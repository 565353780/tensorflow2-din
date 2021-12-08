# din-tf2

## Install
```bash
conda create -n tf python=3.7
conda activate tf
pip install tensorflow-gpu tqdm
ln -s <your-datasets-folder-path> ./datasets
```

## Run
```bash
cd DIEN
python train.py
```

## Visual
```bash
tensorboard --logdir ./logs --host 0.0.0.0
```

## Enjoy it~

