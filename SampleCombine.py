import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import tensorflow as tf
from tensorflow import keras as kr
from tensorflow.keras.preprocessing import image as img
import numpy as np
def load(x):
    return img.img_to_array(img.load_img(x))

path_end = '/results/samples/real_rec_images/real_rec_29.png'
base_path = 'FFHQ/' 

mse = load(base_path + 'MSE' + path_end)
info = load(base_path + 'InfoGAN' + path_end)
dls = load(base_path + 'DLSGAN' + path_end)
pvd = load(base_path + 'PVDGAN' + path_end)
vae_p0 = load(base_path + 'VAEGAN_P0' + path_end)
vae_p1 = load(base_path + 'VAEGAN_P1' + path_end)
vae_p10 = load(base_path + 'VAEGAN_P10' + path_end)

res=256

mses = []
infos = []
dlss = []
pvds = []
reals = []
vae_p0s = []
vae_p1s = []
vae_p10s = []

for i in range(1):
    k = 0
    mses.append(        mse[:,k*(res*2+5)+res:k*(res*2+5)+2*res])
    infos.append(      info[:,k*(res*2+5)+res:k*(res*2+5)+2*res])
    dlss.append(        dls[:,k*(res*2+5)+res:k*(res*2+5)+2*res])
    pvds.append(        pvd[:,k*(res*2+5)+res:k*(res*2+5)+2*res])
    reals.append(       mse[:,k*(res*2+5)    :k*(res*2+5)+  res])
    vae_p0s.append(  vae_p0[:,k*(res*2+5)+res:k*(res*2+5)+2*res])
    vae_p1s.append(  vae_p1[:,k*(res*2+5)+res:k*(res*2+5)+2*res])
    vae_p10s.append(vae_p10[:,k*(res*2+5)+res:k*(res*2+5)+2*res])

blocks = []
for mse_, info_, dls_, pvd_, real_, vae_p0_, vae_p1_, vae_p10_ in zip(mses, infos, dlss, pvds, reals, vae_p0s, vae_p1s, vae_p10s):
    blocks.append(mse_)
    blocks.append(info_)
    blocks.append(dls_)
    blocks.append(pvd_)
    blocks.append(real_)
    blocks.append(vae_p0_)
    blocks.append(vae_p1_)
    blocks.append(vae_p10_)

img.save_img('real_rec.png', np.hstack(blocks))
