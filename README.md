# din-tf2

## Download
```bash
git clone ssh://git@chLi:30001/mine/din-tf2.git
cd din-tf2

```

## Prepare datasets
```bash
# windows
mklink /D <path-to-your-datasets-folder> ./datasets
# linux
ln -s <path-to-your-datasets-folder> ./datasets
```

### NOTE
```bash
Your din-tf2/ folder struct must looks like this:

din-tf2/
|
|---AFM/
|   |---model.py
|   |---train.py
|   |---...
|
|---datasets/
|   |---Criteo/
|   |---raw_data/
|   |---build_dataset.py
|   |---...
|
|---DIEN/
|   |---afm.py
|   |---data.py
|   |---DINTrainer.py
|   |---layer.py
|   |---model.py
|   |---train.py
|   |---utils.py
|   |---...
|
|---DIN/
|   |---DIN_TF1/
|   |---Dice.py
|   |---...
|
|---utils/
|   |---1_convert_pd.py
|   |---2_remap_id.py
|   |---...
|
|---README.md
```

## Install
```bash
conda create -n tf python=3.7
conda activate tf
pip install tensorflow-gpu tqdm pandas
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

