# Temporally-Aware Feature Pooling for Dense Video Captioning in Video Broadcasts

This the code for the paper [SoccerNet-Caption: Dense Video Captioning for Soccer Broadcasts Commentaries](https://arxiv.org/pdf/2304.04565.pdf) (CVSports2023). The training is divided in two phase : spotting training phase and captioning training phase.

## Create Environment

```bash
conda create -y -n DVC-AdvancedPooling python=3.8
conda activate DVC-AdvancedPooling
conda install -y pytorch=1.13 torchvision=0.7 cudatoolkit=10.1 torchtext=0.14 -c pytorch
pip install SoccerNet matplotlib sklearn spacy
pip install git+https://github.com/Maluuba/nlg-eval.git@master
python -m spacy download en_core_web_sm
```

## Train NetVLAD++

`python src/spotting.py --SoccerNet_path=path/to/SoccerNet/ --model_name=NetVLAD++`

once the spotting training is done you can start the captioning training with

`python src/captioning.py --SoccerNet_path=path/to/SoccerNet/ --model_name=NetVLAD++`

Replace `path/to/SoccerNet/` with a local path for the SoccerNet dataset. If you do not have a copy of SoccerNet, this code will automatically download SoccerNet.

## Inference

```bash
python src/spotting.py --SoccerNet_path=path/to/SoccerNet/ --model_name=NetVLAD++ --test_only
python src/captioning.py --SoccerNet_path=path/to/SoccerNet/ --model_name=NetVLAD++ --test_only
```

## Submission for the SoccerNet-v2 Challenge

For the SoccerNet-v2 challenge, we train on an aggregation of the train+val sets, we validate on the test set and infer on the challenge set.

```bash
python src/spotting.py --SoccerNet_path=path/to/SoccerNet/ --model_name=NetVLAD++_Challenge --split_train train valid --split_valid test --split_test challenge
python src/captioning.py --SoccerNet_path=path/to/SoccerNet/ --model_name=NetVLAD++_Challenge --split_train train valid --split_valid test --split_test challenge
```

### More encoders

SoccerNet-v2 provide a list of alternative video frame features:

- `--features=ResNET_TF2.npy`: ResNET features from SoccerNet-v2
- `--features=ResNET_TF2_PCA512.npy`: ResNET features from SoccerNet-v2 reduced at dimension 512 with PCA
- `--features=C3D.npy`: C3D features from SoccerNet
- `--features=C3D_PCA512.npy`: C3D features from SoccerNet reduced at dimension 512 with PCA
- `--features=I3D.npy`: I3D features from SoccerNet
- `--features=I3D_PCA512.npy`: I3D features from SoccerNet reduced at dimension 512 with PCA
- `--features=baidu_soccer_embeddings.npy:`: Baidu's winner team features for SoccerNet-v2

### More temporally-aware pooling modules

We developed alternative pooling module

- `--pool=NetVLAD`: NetVLAD pooling module
- `--pool=NetVLAD++`: Temporally aware NetVLAD pooling module
- `--pool=NetRVLAD`: NetRVLAD pooling module
- `--pool=NetRVLAD++`: Temporally aware NetRVLAD pooling module