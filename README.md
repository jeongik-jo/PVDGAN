# PVDGAN

Official implementation of PVDGAN  
  
Environment: Windows 11 WSL2, Tensorflow 2.15.0.post1

To train the model:
1. Download FFHQ images from https://github.com/NVlabs/ffhq-dataset and AFHQ images from https://www.kaggle.com/datasets/andrewmvd/animal-faces
2. Put images to "dataset/ffhq/train", "dataset/ffhq/test", "dataset/afhq/train", "dataset/afhq/test" folders.
3. Set Hyperparameters.py, then run Main.py

Algorithm is in "Train.py"
