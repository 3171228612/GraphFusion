import torch
import torch.nn as nn
from .aggregator import ConvBlock

class CrossAttentionLayer(nn.Module):

    def __init__(self, d_model=256, nhead=8, dropout=0.0):
        super().__init__()

        self.attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def with_pos_embed(self, tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward(self, source, query, batch_offsets, attn_masks=None, pe=None):
        """
        source (B*N, d_model)
        batch_offsets List[int] (b+1)
        query Tensor (b, n_q, d_model)
        """
        B = len(batch_offsets) - 1
        outputs = []
        query = self.with_pos_embed(query, pe)
        for i in range(B):
            start_id = batch_offsets[i]
            end_id = batch_offsets[i + 1]
            k = v = source[start_id:end_id].unsqueeze(0)  # (1, n, d_model)
            if attn_masks:
                output, _ = self.attn(query[i].unsqueeze(0), k, v, attn_mask=attn_masks[i])  # (1, 100, d_model)
            else:
                output, _ = self.attn(query[i].unsqueeze(0), k, v)
            self.dropout(output)
            output = output + query[i]
            self.norm(output)
            outputs.append(output)
        outputs = torch.cat(outputs, dim=0)  # (b, 100, d_model)
        return outputs


class SelfAttentionLayer(nn.Module):

    def __init__(self, d_model=256, nhead=8, dropout=0.0):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            d_model,
            nhead,
            dropout=dropout,
            batch_first=True,
        )
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def with_pos_embed(self, tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward(self, x, pe=None):
        """
        x Tensor (b, 100, c)
        """
        q = k = self.with_pos_embed(x, pe)
        output, _ = self.attn(q, k, x)
        output = self.dropout(output) + x
        output = self.norm(output)
        return output

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=1)
        self.bn = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x



class FFN(nn.Module):

    def __init__(self, d_model, hidden_dim, dropout=0.0, activation_fn='relu'):
        super().__init__()
        if activation_fn == 'relu':
            self.net = nn.Sequential(
                nn.Linear(d_model, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, d_model),
                nn.Dropout(dropout),
            )
        elif activation_fn == 'gelu':
            self.net = nn.Sequential(
                nn.Linear(d_model, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, d_model),
                nn.Dropout(dropout),
            )
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        output = self.net(x)
        output = output + x
        output = self.norm(output)
        return output


class QueryDecoder(nn.Module):
    """
    in_channels List[int] (4,) [64,96,128,160]
    """

    def __init__(
        self,
        num_layer=6,
        num_query=100,
        num_class=18,
        in_channel=32,
        d_model=256,
        nhead=8,
        hidden_dim=1024,
        dropout=0.0,
        activation_fn='relu',
        iter_pred=False,
        attn_mask=False,
        pe=False,
    ):
        super().__init__()
        self.num_layer = num_layer
        self.num_query = num_query
        self.input_proj = nn.Sequential(nn.Linear(in_channel, d_model), nn.LayerNorm(d_model), nn.ReLU())
        #self.input_feat = nn.Sequential(nn.Linear(hidden_dim*2, d_model), nn.LayerNorm(d_model), nn.ReLU())
        self.query = nn.Embedding(num_query, d_model)
        self.query1 = nn.Embedding(num_query, d_model)
        if pe:
            self.pe = nn.Embedding(num_query, d_model)
        self.cross_attn_layers = nn.ModuleList([])
        self.self_attn_layers = nn.ModuleList([])
        self.ffn_layers = nn.ModuleList([])
        self.query_layers = nn.Sequential(nn.Linear(in_channel*2, d_model), nn.LayerNorm(d_model), nn.ReLU())
        self.fc = nn.Linear(d_model*2,d_model)
        self.mask_tower = nn.Sequential(
            ConvBlock(d_model, d_model),
            ConvBlock(d_model, d_model),
            ConvBlock(d_model, d_model),
            nn.Conv1d(d_model, d_model, kernel_size=1)
        )


        for i in range(num_layer):
            self.cross_attn_layers.append(CrossAttentionLayer(d_model, nhead, dropout))
            self.self_attn_layers.append(SelfAttentionLayer(d_model, nhead, dropout))
            self.ffn_layers.append(FFN(d_model, hidden_dim, dropout, activation_fn))
        self.out_norm = nn.LayerNorm(d_model)
        self.out_cls = nn.Sequential(nn.Linear(d_model, d_model), nn.ReLU(), nn.Linear(d_model, num_class + 1))
        self.out_score = nn.Sequential(nn.Linear(d_model, d_model), nn.ReLU(), nn.Linear(d_model, 1))
        self.x_mask = nn.Sequential(nn.Linear(in_channel, d_model), nn.ReLU(), nn.Linear(d_model, d_model))
        self.iter_pred = iter_pred
        self.attn_mask = attn_mask
        

    def get_mask(self, query, mask_feats, batch_offsets):
        pred_masks = []
        attn_masks = []
        for i in range(len(batch_offsets) - 1):
            start_id, end_id = batch_offsets[i], batch_offsets[i + 1]
            mask_feat = mask_feats[start_id:end_id]
            pred_mask = torch.einsum('nd,md->nm', query[i], mask_feat)
            if self.attn_mask:
                attn_mask = (pred_mask.sigmoid() < 0.5).bool()
                attn_mask[torch.where(attn_mask.sum(-1) == attn_mask.shape[-1])] = False
                attn_mask = attn_mask.detach()
                attn_masks.append(attn_mask)
            pred_masks.append(pred_mask)

        return pred_masks, attn_masks

    def prediction_head(self, query, query_feat, mask_feats, batch_offsets, offsets):
        try:
            query = self.out_norm(query)
            pred_labels = self.out_cls(query)
            pred_scores = self.out_score(query)
            _, attn_masks1 = self.get_mask(query, query_feat, offsets)
            pred_masks, attn_masks = self.get_mask(query, mask_feats, batch_offsets)

            return pred_labels, pred_scores, pred_masks, attn_masks, attn_masks1
        except:
            print('baocuole',query.shape,mask_feats.shape,batch_offsets)
    def forward_simple(self, x, batch_offsets):
        inst_feats = self.input_proj(x)
        mask_feats = self.x_mask(x)
        B = len(batch_offsets) - 1
        query = self.query.weight.unsqueeze(0).repeat(B, 1, 1)  # (b, n, d_model)
        for i in range(self.num_layer):
            query = self.cross_attn_layers[i](inst_feats, query, batch_offsets)
            query = self.self_attn_layers[i](query)
            query = self.ffn_layers[i](query)
        pred_labels, pred_scores, pred_masks, _ = self.prediction_head(query, mask_feats, batch_offsets)
        return {'labels': pred_labels, 'masks': pred_masks, 'scores': pred_scores}

    def weighted_sum(self, query1, query2, weights=None):
        if weights is None:
            # 如果没有提供权重，将权重设置为1
            weights = [1, 1]
        # 确保权重是一个长度为2的列表
        assert len(weights) == 2, "weights should be a list of length 2"

        # 将权重转换为张量，并调整形状以匹配查询向量
        weight1 = torch.full_like(query1, weights[0])
        weight2 = torch.full_like(query2, weights[1])

        weighted_query1 = weight1 * query1
        weighted_query2 = weight2 * query2
        return weighted_query1 + weighted_query2

    def create_offsets(self, num_query, num_batches):
        return [i * num_query for i in range(num_batches+1)]

    def forward_iter_pred(self, x, query_feat, batch_offsets):
        """
        x [B*M, inchannel]
        """
        prediction_labels = []
        prediction_masks = []
        prediction_scores = []
        query_feat= self.query_layers(query_feat)
        inst_feats = self.input_proj(x)
        mask_feats = self.x_mask(x)

        query_feat = self.mask_tower(query_feat.unsqueeze(0).permute(0,2,1)).permute(0,2,1).squeeze(0)
        B = len(batch_offsets) - 1
        query = self.query.weight.unsqueeze(0).repeat(B, 1, 1)  # (b, n, d_model)
        query1 = self.query.weight.unsqueeze(0).repeat(B, 1, 1)  # (b, n, d_model)
        offsets = self.create_offsets(self.num_query,B)
       

        if getattr(self, 'pe', None):
            pe = self.pe.weight.unsqueeze(0).repeat(B, 1, 1)
        else:
            pe = None
        
        pred_labels, pred_scores, pred_masks, attn_masks, attn_masks1 = self.prediction_head(query, query_feat, mask_feats, batch_offsets, offsets)
        prediction_labels.append(pred_labels)
        prediction_scores.append(pred_scores)
        prediction_masks.append(pred_masks)
        for i in range(self.num_layer):
            query1 = self.cross_attn_layers[i](query_feat, query1, offsets, attn_masks1, pe)
            query = self.cross_attn_layers[i](inst_feats, query, batch_offsets, attn_masks, pe)
            query1 = self.self_attn_layers[i](query1, pe)
            query = self.self_attn_layers[i](query, pe)
            query1 = self.ffn_layers[i](query1)
            query = self.ffn_layers[i](query)

            #query = self.self_attn_layers[i](query, query1)
            #query = self.ffn_layers[i](query)
            
            query_combined = torch.cat([query, query1], dim=2)
            
            #query = self.mask_tower(query_combined.permute(0,2,1)).permute(0,2,1)
            #query = self.mask_tower(query_combined)
            query = self.fc(query_combined)

            pred_labels, pred_scores, pred_masks, attn_masks, attn_masks1 = self.prediction_head(query, query_feat, mask_feats, batch_offsets, offsets)
            prediction_labels.append(pred_labels)
            prediction_scores.append(pred_scores)
            prediction_masks.append(pred_masks)
        return {
            'labels':
            pred_labels,
            'masks':
            pred_masks,
            'scores':
            pred_scores,
            'aux_outputs': [{
                'labels': a,
                'masks': b,
                'scores': c
            } for a, b, c in zip(
                prediction_labels[:-1],
                prediction_masks[:-1],
                prediction_scores[:-1],
            )],
        }

    def forward(self, x,query_feat, batch_offsets):
        if self.iter_pred:
            return self.forward_iter_pred(x,query_feat, batch_offsets)
        else:
            return self.forward_simple(x, batch_offsets)
