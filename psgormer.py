import functools
import gorilla
import pointgroup_ops
import spconv.pytorch as spconv
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter_max, scatter_mean

from psgormer.utils import cuda_cast, rle_encode
from .backbone import ResidualBlock, UBlock
from .loss import Criterion
from .query_decoder import QueryDecoder
from .aggregator import LocalAggregator, MLP

@gorilla.MODELS.register_module()
class PSGformer(nn.Module):

    def __init__(
        self,
        input_channel: int = 6,
        blocks: int = 5,
        block_reps: int = 2,
        media: int = 32,
        normalize_before=True,
        return_blocks=True,
        instance_head_cfg=None,
        pool='mean',
        num_class=18,
        channels=32,
        num_query=400,
        filter_bg_thresh=0.1,
        decoder=None,
        criterion=None,
        test_cfg=None,
        norm_eval=False,
        fix_module=[],
    ):
        super().__init__()

        self.instance_head_cfg = instance_head_cfg
        self.channels = channels

        # backbone and pooling
        self.input_conv = spconv.SparseSequential(
            spconv.SubMConv3d(
                input_channel,
                media,
                kernel_size=3,
                padding=1,
                bias=False,
                indice_key='subm1',
            ))
        block = ResidualBlock
        norm_fn = functools.partial(nn.BatchNorm1d, eps=1e-4, momentum=0.1)
        block_list = [media * (i + 1) for i in range(blocks)]
        self.unet = UBlock(
            block_list,
            norm_fn,
            block_reps,
            block,
            indice_key_id=1,
            normalize_before=normalize_before,
            return_blocks=return_blocks,
        )
        self.output_layer = spconv.SparseSequential(norm_fn(media), nn.ReLU(inplace=True))
        self.pool = pool
        self.num_class = num_class

        #new
        self.filter_bg_thresh = filter_bg_thresh
        # # NOTE point-wise prediction
        self.semantic_linear = MLP(channels, num_class+2, norm_fn=norm_fn, num_layers=2)

        self.aggregator = LocalAggregator(
                mlp_dim=self.channels,
                n_sample=instance_head_cfg.n_sample_pa1,
                radius=0.2 * instance_head_cfg.radius_scale,
                n_neighbor=instance_head_cfg.neighbor,
                n_neighbor_post=instance_head_cfg.neighbor * 2,
            )

        # decoder
        self.decoder = QueryDecoder(**decoder, in_channel=media, num_class=num_class)

        # criterion
        self.criterion = Criterion(**criterion, num_class=num_class)

        self.test_cfg = test_cfg
        self.norm_eval = norm_eval
        for module in fix_module:
            module = getattr(self, module)
            module.eval()
            for param in module.parameters():
                param.requires_grad = False

    def train(self, mode=True):
        super(PSGormer, self).train(mode)
        if mode and self.norm_eval:
            for m in self.modules():
                # trick: eval have effect on BatchNorm1d only
                if isinstance(m, nn.BatchNorm1d):
                    m.eval()

    def forward(self, batch, mode='loss'):
        if mode == 'loss':
            return self.loss(**batch)
        elif mode == 'predict':
            return self.predict(**batch)

    @cuda_cast
    def loss(self, scan_ids, voxel_coords, p2v_map, v2p_map, spatial_shape, feats, coords_float, insts, superpoints, batch_offsets):
        
        batch_size = len(batch_offsets) - 1
        voxel_feats = pointgroup_ops.voxelization(feats, v2p_map)
        input = spconv.SparseConvTensor(voxel_feats, voxel_coords.int(), spatial_shape, batch_size)

        voxel_output_feats, voxel_feat, output_batch_idxs = self.extract_feat(input, superpoints, p2v_map)

        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        voxel_spps = superpoints[v2p_map[:, 1].long()]
        voxel_spps = torch.unique(voxel_spps, return_inverse=True)[1]

        (voxel_batch_idxs_,voxel_output_feats_,voxel_coords_float_,voxel_batch_offsets_) = \
            self.IA_aware(voxel_feat, output_batch_idxs, coords_float, voxel_spps, batch_size)

        query_feat, idx, query_coor = self.aggregator(
            voxel_coords_float_,
            voxel_output_feats_,
            batch_offsets=voxel_batch_offsets_,
            batch_size=batch_size,
            sampled_before=False,
        ) #B*n_neighbor , mlp

        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        out = self.decoder(voxel_output_feats, query_feat, batch_offsets)

        loss, loss_dict = self.criterion(out, insts)
        return loss, loss_dict

    @cuda_cast
    def predict(self, scan_ids, voxel_coords, p2v_map, v2p_map, spatial_shape, feats, coords_float, insts, superpoints, batch_offsets):
        batch_size = len(batch_offsets) - 1
        voxel_feats = pointgroup_ops.voxelization(feats, v2p_map)
        input = spconv.SparseConvTensor(voxel_feats, voxel_coords.int(), spatial_shape, batch_size)

        voxel_output_feats, voxel_feat, output_batch_idxs = self.extract_feat(input, superpoints,
                                                                              p2v_map)

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        voxel_spps = superpoints[v2p_map[:, 1].long()]
        voxel_spps = torch.unique(voxel_spps, return_inverse=True)[1]

        (voxel_batch_idxs_, voxel_output_feats_, voxel_coords_float_, voxel_batch_offsets_) = \
            self.IA_aware(voxel_feat, output_batch_idxs, coords_float, voxel_spps, batch_size)

        query_feat, idx, query_coor = self.aggregator(
            voxel_coords_float_,
            voxel_output_feats_,
            batch_offsets=voxel_batch_offsets_,
            batch_size=batch_size,
            sampled_before=False,
        )  # B*n_neighbor , mlp

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        out = self.decoder(voxel_output_feats, query_feat, batch_offsets)

        ret = self.predict_by_feat(scan_ids, out, superpoints, insts)
        return ret

    def predict_by_feat(self, scan_ids, out, superpoints, insts):
        pred_labels = out['labels']
        pred_masks = out['masks']
        pred_scores = out['scores']

        scores = F.softmax(pred_labels[0], dim=-1)[:, :-1]
        scores *= pred_scores[0]
        labels = torch.arange(
            self.num_class, device=scores.device).unsqueeze(0).repeat(self.decoder.num_query, 1).flatten(0, 1)
        scores, topk_idx = scores.flatten(0, 1).topk(self.test_cfg.topk_insts, sorted=False)
        labels = labels[topk_idx]
        labels += 1

        topk_idx = torch.div(topk_idx, self.num_class, rounding_mode='floor')
        mask_pred = pred_masks[0]
        mask_pred = mask_pred[topk_idx]
        mask_pred_sigmoid = mask_pred.sigmoid()
        # mask_pred before sigmoid()
        mask_pred = (mask_pred > 0).float()  # [n_p, M]
        mask_scores = (mask_pred_sigmoid * mask_pred).sum(1) / (mask_pred.sum(1) + 1e-6)
        scores = scores * mask_scores
        # get mask
        mask_pred = mask_pred[:, superpoints].int()

        # score_thr
        score_mask = scores > self.test_cfg.score_thr
        scores = scores[score_mask]  # (n_p,)
        labels = labels[score_mask]  # (n_p,)
        mask_pred = mask_pred[score_mask]  # (n_p, N)

        # npoint thr
        mask_pointnum = mask_pred.sum(1)
        npoint_mask = mask_pointnum > self.test_cfg.npoint_thr
        scores = scores[npoint_mask]  # (n_p,)
        labels = labels[npoint_mask]  # (n_p,)
        mask_pred = mask_pred[npoint_mask]  # (n_p, N)

        cls_pred = labels.cpu().numpy()
        score_pred = scores.cpu().numpy()
        mask_pred = mask_pred.cpu().numpy()

        pred_instances = []
        for i in range(cls_pred.shape[0]):
            pred = {}
            pred['scan_id'] = scan_ids[0]
            pred['label_id'] = cls_pred[i]
            pred['conf'] = score_pred[i]
            # rle encode mask to save memory
            pred['pred_mask'] = rle_encode(mask_pred[i])
            pred_instances.append(pred)

        gt_instances = insts[0].gt_instances
        return dict(scan_id=scan_ids[0], pred_instances=pred_instances, gt_instances=gt_instances)

    def get_batch_offsets(self,batch_idxs, bs):
        batch_offsets = torch.zeros((bs + 1), dtype=torch.int, device=batch_idxs.device)
        for i in range(bs):
            batch_offsets[i + 1] = batch_offsets[i] + (batch_idxs == i).sum()
        assert batch_offsets[-1] == batch_idxs.shape[0]
        return batch_offsets

    def IA_aware(self, output_feats, batch_idxs, coords_float, voxel_spps, batch_size):

        semantic_scores = self.semantic_linear(output_feats)

        voxel_semantic_scores_sm = F.softmax(semantic_scores, dim=1)
        spp_semantic_scores_sm = self.custom_scatter_mean(
            voxel_semantic_scores_sm, voxel_spps, dim=0, pool=True
        )
        spp_object_conditions = torch.any(
            spp_semantic_scores_sm[:, 2:] >= self.filter_bg_thresh, dim=-1
        )
        object_conditions = spp_object_conditions[voxel_spps]
        object_idxs = torch.nonzero(object_conditions).view(-1)
        if len(object_idxs) <= 100:
            loss_dict = {
                "placeholder": torch.tensor(0.0, requires_grad=True, device='cuda', dtype=torch.float)
            }
            return self.parse_losses(loss_dict)
        voxel_batch_idxs_ = batch_idxs[object_idxs]
        voxel_output_feats_ = output_feats[object_idxs]
        voxel_coords_float_ = coords_float[object_idxs]
        voxel_batch_offsets_ = self.get_batch_offsets(voxel_batch_idxs_, batch_size)
        return  voxel_batch_idxs_,voxel_output_feats_,voxel_coords_float_,voxel_batch_offsets_

    def custom_scatter_mean(self,input_feats, indices, dim=0, pool=True, output_type=None):

        if not pool:
            return input_feats

        original_type = input_feats.dtype
        with torch.cuda.amp.autocast(enabled=False): 
            out_feats = scatter_mean(input_feats.to(torch.float32), indices, dim=dim)

        if output_type is None:  
            out_feats = out_feats.to(original_type)
        else:
            out_feats = out_feats.to(output_type)

        return out_feats

    def extract_feat(self, x, superpoints, v2p_map):
        # backbone
        x = self.input_conv(x)
        x, _ = self.unet(x)
        x = self.output_layer(x)
        voxel_feat = x.features
        output_batch_idxs = x.indices[:, 0]
        x = x.features[v2p_map.long()]  # (B*N, media)

        # superpoint pooling
        if self.pool == 'mean':
            x = scatter_mean(x, superpoints, dim=0)  # (B*M, media)
        elif self.pool == 'max':
            x, _ = scatter_max(x, superpoints, dim=0)  # (B*M, media)


        return x, voxel_feat, output_batch_idxs
