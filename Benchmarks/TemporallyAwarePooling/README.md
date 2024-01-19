# Temporally-Aware Feature Pooling for Dense Video Captioning in Video Broadcasts

This the code for the paper [SoccerNet-Caption: Dense Video Captioning for Soccer Broadcasts Commentaries](https://arxiv.org/pdf/2304.04565.pdf) (CVSports2023). The training is divided in two phase : spotting training phase and captioning training phase.

## Create Environment

```bash
conda create -y -n soccernet-DVC python=3.8
conda activate soccernet-DVC
conda install -y pytorch torchvision torchtext pytorch-cuda -c pytorch -c nvidia
pip install SoccerNet matplotlib scikit-learn spacy wandb
pip install git+https://github.com/Maluuba/nlg-eval.git@master
python -m spacy download en_core_web_sm
pip install torchtext
```

## Download weights

```
mkdir models
```

[Download](https://drive.google.com/file/d/107f4dpd6ooJZ-gcjXkJZc1Lw7Qq7CD4r/view?usp=share_link) and extract in the folder models

## Train the model

```
python src/main.py --SoccerNet_path=path/to/SoccerNet/ --model_name=new_model --features=baidu_soccer_embeddings.npy --framerate 1 --pool=NetVLAD --window_size_caption 45 --window_size_spotting 15 --NMS_window 30 --num_layers 4 --first_stage caption --pretrain --GPU 0
```

Replace `path/to/SoccerNet/` with a local path for the SoccerNet dataset. If you do not have a copy of SoccerNet, this code will automatically download SoccerNet.

## Inference

```bash
python src/main.py --SoccerNet_path=path/to/SoccerNet/ --model_name=baidu-NetVLAD-pretrain-caption --features=baidu_soccer_embeddings.npy --framerate 1 --pool=NetVLAD --window_size_caption 45 --window_size_spotting 15 --NMS_window 30 --num_layers 4 --first_stage caption --pretrain --GPU 0 --test_only
```

### More encoders

SoccerNet-v3 provide a list of alternative video frame features:

- `--features=ResNET_TF2.npy`: ResNET features from SoccerNet-v2
- `--features=ResNET_TF2_PCA512.npy`: ResNET features from SoccerNet-v2 reduced at dimension 512 with PCA
- `--features=C3D.npy`: C3D features from SoccerNet
- `--features=C3D_PCA512.npy`: C3D features from SoccerNet reduced at dimension 512 with PCA
- `--features=I3D.npy`: I3D features from SoccerNet
- `--features=I3D_PCA512.npy`: I3D features from SoccerNet reduced at dimension 512 with PCA
- `--features=baidu_soccer_embeddings.npy:`: Baidu's winner team features for SoccerNet-v2

### More aggregator modules

We developed alternative pooling module

- `--pool=NetVLAD`: NetVLAD pooling module
- `--pool=NetVLAD++`: Temporally aware NetVLAD pooling module
- `--pool=NetRVLAD`: NetRVLAD pooling module
- `--pool=NetRVLAD++`: Temporally aware NetRVLAD pooling module


### More training procedure

- `--first_stage caption --freeze_encoder`: Train first on captioning, and then frozen weights are transferred to spotting
- `--first_stage spotting --freeze_encoder`: Train first on spotting, and then frozen weights are transferred to captioning
- `--first_stage caption --pretrain`: Train first on captioning, and then weights are transferred to spotting and fine-tuned
- `--first_stage spotting --pretrain`: Train first on spotting, and then weights are transferred to captioning and fine-tuned
- ``: Both submodels are trained from scratch independently