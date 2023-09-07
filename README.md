
## Installation

Requirements

- Python 3.x
- Pytorch 1.10
- CUDA 10.x or higher

The following installation suppose `python=3.8` `pytorch=1.10` and `cuda=11.3`.

- Create a conda virtual environment

  ```
  conda create -n psgformer python=3.8
  conda activate psgformer
  ```

- Install the dependencies

  ```
  pip install spconv-cu113
  conda install pytorch-scatter -c pyg (test on the 2.0.9 version)
  pip install -r requirements.txt
  ```

  Install  [segmentator](https://github.com/Karbo123/segmentator) (Then wrap the segmentator in ScanNet).
  ```
  git clone https://github.com/Karbo123/segmentator.git

  cd segmentator/csrc
  mkdir build && cd build

  cmake .. \
  -DCMAKE_PREFIX_PATH=`python -c 'import torch;print(torch.utils.cmake_prefix_path)'` \
  -DPYTHON_INCLUDE_DIR=$(python -c "from distutils.sysconfig import get_python_inc; print(get_python_inc())")  \
  -DPYTHON_LIBRARY=$(python -c "import distutils.sysconfig as sysconfig; print(sysconfig.get_config_var('LIBDIR'))") \
  -DCMAKE_INSTALL_PREFIX=`python -c 'from distutils.sysconfig import get_python_lib; print(get_python_lib())'` 

  make && make install
  ```
  Setup pointnet2
  ```
  cd psgformer/pointnet2
  python3 setup.py bdist_wheel
  cd ./dist
  pip3 install <.whl>
  ```
- Setup, Install psgformer and pointgroup_ops.

  ```
  sudo apt-get install libsparsehash-dev
  python setup.py develop
  cd psgformer/lib/
  python setup.py build_ext develop
  ```

## Data Preparation

### ScanNet v2 dataset

Download the [ScanNet](http://www.scan-net.org/) v2 dataset.

Put the downloaded `scans` and `scans_test` folder as follows.

```
SPFormer
├── data
│   ├── scannetv2
│   │   ├── scans
│   │   ├── scans_test
```

Split and preprocess data

```
cd data/scannetv2
bash prepare_data.sh
```

The script data into train/val/test folder and preprocess the data. After running the script the scannet dataset structure should look like below.

```
SPFormer
├── data
│   ├── scannetv2
│   │   ├── scans
│   │   ├── scans_test
│   │   ├── train
│   │   ├── val
│   │   ├── test
│   │   ├── val_gt
```

## Pretrained Model

Download [SSTNet](https://drive.google.com/file/d/1vucwdbm6pHRGlUZAYFdK9JmnPVerjNuD/view?usp=sharing) pretrained model (We only use the Sparse 3D U-Net backbone for training).

Move the pretrained model to checkpoints.

```
mkdir checkpoints
mv ${Download_PATH}/sstnet_pretrain.pth checkpoints/
```

## Training

```
python tools/train.py configs/psg_scannet.yaml
```

## Inference

For evaluation on ScanNetV2 val

We have already put the pre-training model under the folder

```
python tools/test.py configs/psg_scannet.yaml checkpoints/psg_scannet_512.pth
```


## Ancknowledgement

Sincerely thanks for [SoftGroup](https://github.com/thangvubk/SoftGroup) and [SSTNet](https://github.com/Gorilla-Lab-SCUT/SSTNet) repos. This repo is build upon them.

