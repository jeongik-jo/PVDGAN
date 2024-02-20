import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import tensorflow as tf
from tensorflow import keras as kr

enc_opt = kr.optimizers.Adam(learning_rate=0.003, beta_1=0.0, beta_2=0.99,
                             use_ema=True, ema_momentum=0.999, ema_overwrite_frequency=None)
dec_opt = kr.optimizers.Adam(learning_rate=0.003, beta_1=0.0, beta_2=0.99,
                             use_ema=True, ema_momentum=0.999, ema_overwrite_frequency=None)

is_ffhq = False
img_res = 256
img_chn = 3
ltn_dim = 1024

reg_w = 3.0
img_rec_w = 1.0
rec_is_perceptual = True
is_dls = True
ltn_var_trace = tf.Variable(tf.ones([ltn_dim]) * 0.0001, name='ltn_var_trace', trainable=False)
if is_dls:
    ltn_rec_w = 1.0
    use_logvar = False
    is_pvd = True
    ltn_var_decay_rate = 0.999
else:
    prr_w = 1.0
    dis_opt = kr.optimizers.Adam(learning_rate=0.003, beta_1=0.0, beta_2=0.99)

batch_size = 8

train_data_size = -1
test_data_size = -1
epochs = 100

load_model = False

eval_model = True
epoch_per_eval = 1


def ltn_dist_func(batch_size):
    return tf.random.normal([batch_size, ltn_dim])


def get_ltn_scl_vecs():
    return tf.sqrt(tf.cast(ltn_dim, 'float32') * ltn_var_trace / tf.reduce_sum(ltn_var_trace))[tf.newaxis]


def get_ltn_ent():
    return tf.reduce_sum(tf.math.log(get_ltn_scl_vecs() * tf.sqrt(2.0 * 3.141592 * tf.exp(1.0))))


ltn_int_val = 2.0
