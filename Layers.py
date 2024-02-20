import tensorflow as tf
from tensorflow import keras as kr
import HyperParameters as hp


class Conv2D(kr.layers.Layer):
    def __init__(self, filters, kernel_size, padding='SAME', activation=kr.activations.linear, use_bias=True):
        super().__init__()
        self.filters = filters
        self.kernel_size = kernel_size
        self.padding = padding
        self.activation = activation
        self.use_bias = use_bias

    def build(self, input_shape):
        self.multiplier = tf.sqrt(1 / tf.cast(self.kernel_size * self.kernel_size * input_shape[-1], 'float32'))
        self.conv_layer = kr.layers.Conv2D(filters=self.filters, kernel_size=self.kernel_size,
                                           padding=self.padding, use_bias=False, kernel_initializer=kr.initializers.RandomNormal(stddev=1.0))

        if self.use_bias:
            self.bias = tf.Variable(tf.zeros([1, 1, 1, self.filters]), trainable=True, name=self.name + '_bias')

    def call(self, inputs, *args, **kwargs):
        feature_maps = self.conv_layer(inputs) * self.multiplier
        if self.use_bias:
            feature_maps = feature_maps + self.bias
        return self.activation(feature_maps)


class Dense(kr.layers.Layer):
    def __init__(self, units, activation=kr.activations.linear, use_bias=True):
        super().__init__()
        self.units = units
        self.activation = activation
        self.use_bias = use_bias
    def build(self, input_shape):
        self.multiplier = tf.sqrt(1 / tf.cast(input_shape[-1], 'float32'))
        self.dense_layer = kr.layers.Dense(units=self.units, use_bias=False, kernel_initializer=kr.initializers.RandomNormal(stddev=1.0))

        if self.use_bias:
            self.bias = tf.Variable(tf.zeros([1 for _ in range(len(input_shape) - 1)] + [self.units]), trainable=True, name=self.name + '_bias')

    def call(self, inputs, *args, **kwargs):
        feature_vecs = self.dense_layer(inputs) * self.multiplier
        if self.use_bias:
            feature_vecs = feature_vecs + self.bias
        return self.activation(feature_vecs)


class Blur(kr.layers.Layer):
    def __init__(self, upscale=False, downscale=False):
        super().__init__()
        self.upscale = upscale
        self.downscale = downscale

        assert (upscale and downscale) != True

    def build(self, input_shape):
        kernel = tf.cast([1, 3, 3, 1], 'float32')
        kernel = tf.tensordot(kernel, kernel, axes=0)
        kernel = kernel / tf.reduce_sum(kernel)
        self.kernel = tf.tile(kernel[:, :, tf.newaxis, tf.newaxis], [1, 1, input_shape[-1], 1])

        if self.upscale:
            self.w = input_shape[1]
            self.h = input_shape[2]
            self.c = input_shape[3]
            self.kernel = self.kernel * 4
    def call(self, inputs, *args, **kwargs):
        if self.upscale:
            inputs = tf.pad(inputs[:, :, tf.newaxis, :, tf.newaxis, :], [[0, 0], [0, 0], [1, 0], [0, 0], [1, 0], [0, 0]])
            inputs = tf.reshape(inputs, [-1, self.w * 2, self.h * 2, self.c])
            return tf.nn.depthwise_conv2d(input=inputs, filter=self.kernel, strides=[1, 1, 1, 1], padding='SAME')

        elif self.downscale:
            return tf.nn.depthwise_conv2d(input=inputs, filter=self.kernel, strides=[1, 2, 2, 1], padding='SAME')

        else:
            return tf.nn.depthwise_conv2d(input=inputs, filter=self.kernel, strides=[1, 1, 1, 1], padding='SAME')


filter_sizes = [64, 128, 256, 512, 512, 512]
activation = tf.nn.leaky_relu
class Decoder(kr.layers.Layer):
    def __init__(self):
        super().__init__()

    def build(self, input_shape):
        ltn_vec = kr.Input([hp.ltn_dim])
        ftr_maps = kr.layers.Reshape([1, 1, hp.ltn_dim])(ltn_vec)
        ftr_maps = kr.layers.ZeroPadding2D(padding=((3, 3), (3, 3)))(ftr_maps) * 4
        ftr_maps = Conv2D(filters=512, kernel_size=4, padding='VALID', activation=activation)(ftr_maps)

        for filters in reversed(filter_sizes):
            ftr_maps = Blur(upscale=True)(ftr_maps)
            skp_maps = Conv2D(filters=filters, kernel_size=1, use_bias=False)(ftr_maps)
            ftr_maps = Conv2D(filters=filters, kernel_size=3, activation=activation)(ftr_maps)
            ftr_maps = Conv2D(filters=filters, kernel_size=3, activation=activation)(ftr_maps)
            ftr_maps = (skp_maps + ftr_maps) / tf.sqrt(2.0)

        fake_img = Conv2D(filters=3, kernel_size=1)(ftr_maps)

        self.model = kr.Model(ltn_vec, fake_img)

    def call(self, inputs, *args, **kwargs):
        return self.model(inputs, *args, **kwargs)


class Encoder(kr.layers.Layer):
    def __init__(self):
        super().__init__()

    def build(self, input_shape):
        ftr_maps = inp_img = kr.Input([hp.img_res, hp.img_res, hp.img_chn])

        ftr_maps = Conv2D(filters=filter_sizes[0], kernel_size=1, activation=activation)(ftr_maps)
        for i, filters in enumerate(filter_sizes):
            skp_maps = Conv2D(filters=filters, kernel_size=1, use_bias=False)(ftr_maps)
            ftr_maps = Conv2D(filters=filters, kernel_size=3, activation=activation)(ftr_maps)
            ftr_maps = Conv2D(filters=filters, kernel_size=3, activation=activation)(ftr_maps)
            ftr_maps = (ftr_maps + skp_maps) / tf.sqrt(2.0)
            ftr_maps = Blur(downscale=True)(ftr_maps)
        ftr_vec = kr.layers.Flatten()(ftr_maps)
        adv_val = Dense(units=1)(ftr_vec)[:, 0]
        ltn_vec = Dense(units=hp.ltn_dim)(ftr_vec)
        ltn_logvar = Dense(units=hp.ltn_dim)(ftr_vec)
        if hp.rec_is_perceptual:
            rec_ftr_vec = ftr_vec
        else:
            rec_ftr_vec = kr.layers.Flatten()(inp_img)

        self.model = kr.Model(inp_img, [adv_val, ltn_vec, ltn_logvar, rec_ftr_vec])

    def call(self, inputs, *args, **kwargs):
        return self.model(inputs, *args, **kwargs)
