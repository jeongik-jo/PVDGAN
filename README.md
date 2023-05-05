# PVDGAN

Official implementation of PVDGAN  
  
Environment: Windows 11, Tensorflow 2.10  

To train the model:
1. Download FFHQ images from https://github.com/NVlabs/ffhq-dataset
2. Make dataset with SaveDataset.py. Set "folder_path" of SaveDataset.py to path of FFHQ image folder, then run SaveDataset.py
3. Put tfrecord files to './dataset/train' and './dataset/test'.
4. Set Hyperparameters.py, then run Main.py

Algorithm for DLSGAN is in "Train.py"
