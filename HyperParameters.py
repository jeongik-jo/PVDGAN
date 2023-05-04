import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import tensorflow as tf
from tensorflow import keras as kr

enc_opt = kr.optimizers.Adam(learning_rate=0.002, beta_1=0.0, beta_2=0.99)
enc_ema = tf.train.ExponentialMovingAverage(decay=0.999)
dec_opt = kr.optimizers.Adam(learning_rate=0.002, beta_1=0.0, beta_2=0.99)
dec_ema = tf.train.ExponentialMovingAverage(decay=0.999)

image_resolution = 256
latent_dim = 1024

reg_weight = 3.0
image_rec_weight = 1.0
is_dls = True
if is_dls:
    latent_rec_weight = 1.0
    use_image_rec = True
    latent_var_decay_rate = 0.999
else:
    prior_weight = 1.0
    dis_opt = kr.optimizers.Adam(learning_rate=0.002, beta_1=0.0, beta_2=0.99)

batch_size = 8
save_image_size = 8

train_data_size = -1
test_data_size = -1
shuffle_test_dataset = False
epochs = 100

load_model = False

evaluate_model = True
fid_batch_size = batch_size
epoch_per_evaluate = 1


def latent_dist_func(batch_size):
    return tf.random.normal([batch_size, latent_dim])

latent_interpolation_value = 2.0
