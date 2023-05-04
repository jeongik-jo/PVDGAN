import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import tensorflow as tf
from tensorflow import keras as kr
from tensorflow.keras.preprocessing import image as img
import numpy as np
def load(x):
    return img.img_to_array(img.load_img(x))
vae = load(r'VAEGAN/results/samples/real_rec_images/real_rec_99.png')
dls = load(r'DLSGAN/results/samples/real_rec_images/real_rec_99.png')
pvd = load(r'PVDGAN/results/samples/real_rec_images/real_rec_99.png')

res=256

reals = []
vaes = []
pvds = []
dlss = []
ones = []

for i in range(2):
    reals.append(vae[:, i*(res*2+5)      :i*(res*2+5) + res])
    vaes.append(vae[:,  i*(res*2+5) + res:i*(res*2+5) + 2*res])
    pvds.append(pvd[:,  i*(res*2+5) + res:i*(res*2+5) + 2*res])
    dlss.append(dls[:,  i*(res*2+5) + res:i*(res*2+5) + 2*res])
    ones.append(np.ones([res*8, 5, 3]) * 255)

blocks = []
for r, v, d, p, o in zip(reals, vaes, dlss, pvds, ones):
    blocks.append(v)
    blocks.append(r)
    blocks.append(p)
    blocks.append(d)
    blocks.append(o)

img.save_img('real_rec.png', np.hstack(blocks))
