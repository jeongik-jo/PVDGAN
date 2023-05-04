import tensorflow as tf
from tensorflow import keras as kr
import HyperParameters as hp


class Dense(kr.layers.Layer):
    def __init__(self, units, activation=kr.activations.linear, use_bias=True, equalized_lr=False, lr_scale=1.0):
        super(Dense, self).__init__()
        self.units = units
        self.activation = activation
        self.use_bias = use_bias
        self.equalized_lr = equalized_lr
        self.lr_scale = lr_scale

    def build(self, input_shape):
        if self.equalized_lr:
            self.w = tf.Variable(tf.random.normal([input_shape[-1], self.units]) / self.lr_scale, name=self.name + '_w')
            self.he_std = tf.sqrt(tf.cast(input_shape[-1], 'float32'))
        else:
            self.w = tf.Variable(tf.random.normal([input_shape[-1], self.units]) / self.lr_scale /
                                 tf.sqrt(tf.cast(input_shape[-1], 'float32')), name=self.name + '_w')
            self.he_std = 1.0

        if self.use_bias:
            self.b = tf.Variable(tf.zeros([1, self.units]), name=self.name + '_b')

    def call(self, inputs, *args, **kwargs):
        feature_vector = tf.matmul(inputs, self.w) * self.lr_scale / self.he_std
        if self.use_bias:
            feature_vector = feature_vector + self.b

        return self.activation(feature_vector)


class ZeroInsert2D(kr.layers.Layer):
    def __init__(self):
        super(ZeroInsert2D, self).__init__()

    def build(self, input_shape):
        self.reshape_layer = kr.layers.Reshape([input_shape[1], input_shape[2] * 2, input_shape[3] * 2])

    def call(self, inputs, *args, **kwargs):
        feature_maps = tf.stack([tf.zeros_like(inputs), inputs], axis=3)
        feature_maps = tf.stack([tf.zeros_like(feature_maps), feature_maps], axis=5)
        return self.reshape_layer(feature_maps)


class Fir(kr.layers.Layer):
    def __init__(self, kernel, upscale=False, downscale=False):
        super(Fir, self).__init__()
        self.kernel = kernel
        self.upscale = upscale
        self.downscale = downscale

        assert (upscale and downscale) != True
        assert self.kernel.shape[0] == self.kernel.shape[1]

    def build(self, input_shape):
        if self.downscale:
            padding_0 = tf.maximum((self.kernel.shape[0] - 2) // 2, 0)
            padding_1 = tf.maximum(self.kernel.shape[0] - 2 - padding_0, 0)
            self.padding = [[0, 0], [0, 0], [padding_1, padding_0], [padding_1, padding_0]]
        else:
            padding_0 = (self.kernel.shape[0] - 1) // 2
            padding_1 = self.kernel.shape[0] - 1 - padding_0
            self.padding = [[0, 0], [0, 0], [padding_0, padding_1], [padding_0, padding_1]]

        self.kernel = tf.tile(self.kernel[:, :, tf.newaxis, tf.newaxis], [1, 1, input_shape[1], 1])
        if self.upscale:
            self.zero_insert_layer = ZeroInsert2D()
            self.kernel *= 4

    def call(self, inputs, *args, **kwargs):
        if self.upscale:
            return tf.nn.depthwise_conv2d(input=self.zero_insert_layer(inputs), filter=self.kernel, strides=[1, 1, 1, 1],
                                          padding=self.padding, data_format='NCHW')

        elif self.downscale:
            return tf.nn.depthwise_conv2d(input=inputs, filter=self.kernel, strides=[1, 1, 2, 2],
                                          padding=self.padding, data_format='NCHW')

        else:
            return tf.nn.depthwise_conv2d(input=inputs, filter=self.kernel, strides=[1, 1, 1, 1],
                                          padding=self.padding, data_format='NCHW')


def get_blur_kernel():
    kernel = tf.cast([1, 3, 3, 1], 'float32')
    kernel = tf.tensordot(kernel, kernel, axes=0)
    kernel = kernel / tf.reduce_sum(kernel)

    return kernel


class Conv2D(kr.layers.Layer):
    def __init__(self, filters, kernel_size, activation=kr.activations.linear, use_bias=True, equalized_lr=False,
                 upscale=False, downscale=False):
        super(Conv2D, self).__init__()
        self.filters = filters
        self.kernel_size = kernel_size
        self.activation = activation
        self.use_bias = use_bias
        self.equalized_lr = equalized_lr
        self.upscale = upscale
        self.downscale = downscale

        assert (upscale and downscale) != True

    def build(self, input_shape):
        input_filters = input_shape[1]
        if self.equalized_lr:
            self.w = tf.Variable(tf.random.normal([self.kernel_size, self.kernel_size, input_filters, self.filters]),
                                 name=self.name + '_w')
            he_std = tf.sqrt(tf.cast(self.kernel_size * self.kernel_size * input_filters, 'float32'))
        else:
            self.w = tf.Variable(tf.random.normal([self.kernel_size, self.kernel_size, input_filters, self.filters])
                                 / tf.sqrt(tf.cast(self.kernel_size * self.kernel_size * input_filters, 'float32')),
                                 name=self.name + '_w')
            he_std = 1.0

        if self.upscale:
            self.zero_insert_layer = ZeroInsert2D()
            self.blur_layer = Fir(get_blur_kernel() / he_std * 4.0)
            padding_0 = (self.kernel_size - 1) // 2
            padding_1 = self.kernel_size - 1 - padding_0
            self.padding = [[0, 0], [0, 0], [padding_0, padding_1], [padding_0, padding_1]]
        elif self.downscale:
            self.blur_layer = Fir(get_blur_kernel() / he_std)
            padding_0 = tf.maximum((self.kernel_size - 2) // 2, 0)
            padding_1 = tf.maximum(self.kernel_size - 2 - padding_0, 0)
            self.padding = [[0, 0], [0, 0], [padding_1, padding_0], [padding_1, padding_0]]
        else:
            self.he_std = he_std
            padding_0 = (self.kernel_size - 1) // 2
            padding_1 = self.kernel_size - 1 - padding_0
            self.padding = [[0, 0], [0, 0], [padding_0, padding_1], [padding_0, padding_1]]

        if self.use_bias:
            self.b = tf.Variable(tf.zeros([1, self.filters, 1, 1]), name=self.name + '_b')

    def call(self, inputs, *args, **kwargs):
        feature_maps = inputs

        if self.upscale:
            feature_maps = self.zero_insert_layer(inputs)
            feature_maps = tf.nn.conv2d(feature_maps, self.w, strides=1, padding=self.padding, data_format='NCHW')
            feature_maps = self.blur_layer(feature_maps)
        elif self.downscale:
            feature_maps = tf.nn.conv2d(self.blur_layer(feature_maps), self.w, strides=2, padding=self.padding, data_format='NCHW')
        else:
            feature_maps = tf.nn.conv2d(feature_maps, self.w, strides=1, padding=self.padding, data_format='NCHW') / self.he_std

        if self.use_bias:
            feature_maps = feature_maps + self.b

        return self.activation(feature_maps)


filter_sizes = [64, 128, 256, 512, 512, 512]
class Decoder(kr.layers.Layer):
    def __init__(self):
        super(Decoder, self).__init__()

    def build(self, input_shape):
        activation = tf.nn.leaky_relu
        equalized_lr = True

        latent_vector = kr.Input([hp.latent_dim])
        feature_maps = Dense(units=1024 * 4 * 4, activation=activation, equalized_lr=equalized_lr)(latent_vector)
        feature_maps = kr.layers.Reshape([1024, 4, 4])(feature_maps)

        fake_image = Conv2D(filters=3, kernel_size=1, equalized_lr=equalized_lr)(feature_maps)

        for filters in reversed(filter_sizes):
            feature_maps = Conv2D(filters=filters, kernel_size=3, activation=activation, equalized_lr=equalized_lr, upscale=True)(feature_maps)
            feature_maps = Conv2D(filters=filters, kernel_size=3, activation=activation, equalized_lr=equalized_lr)(feature_maps)
            fake_image = Conv2D(filters=3, kernel_size=1, equalized_lr=equalized_lr, use_bias=False)(feature_maps) + \
                         Fir(get_blur_kernel(), upscale=True)(fake_image)

        fake_image = tf.transpose(fake_image, [0, 2, 3, 1])
        self.model = kr.Model(latent_vector, fake_image)

    def call(self, inputs, *args, **kwargs):
        return self.model(inputs, *args, **kwargs)


class Encoder(kr.layers.Layer):
    def __init__(self):
        super(Encoder, self).__init__()

    def build(self, input_shape):
        activation = tf.nn.leaky_relu
        equalized_lr = True

        input_image = kr.Input([hp.image_resolution, hp.image_resolution, 3])
        feature_maps = tf.transpose(input_image, [0, 3, 1, 2])

        feature_maps = Conv2D(filters=filter_sizes[0], kernel_size=1, activation=activation, equalized_lr=equalized_lr)(feature_maps)
        for filters in filter_sizes:
            skip_maps = Conv2D(filters=tf.minimum(filters * 2, 512), kernel_size=1, use_bias=False,
                               equalized_lr=equalized_lr, downscale=True)(feature_maps)
            feature_maps = Conv2D(filters=filters, kernel_size=3, activation=activation, equalized_lr=equalized_lr)(feature_maps)
            feature_maps = Conv2D(filters=tf.minimum(filters * 2, 512), kernel_size=3,
                                  equalized_lr=equalized_lr, downscale=True)(feature_maps)
            feature_maps = activation(feature_maps + skip_maps) / tf.sqrt(2.0)
        feature_vector = kr.layers.Flatten()(feature_maps)

        adv_value = tf.squeeze(Dense(units=1, equalized_lr=equalized_lr)(feature_vector))
        latent_vector = Dense(units=hp.latent_dim, equalized_lr=equalized_lr)(feature_vector)
        latent_logvar = Dense(units=hp.latent_dim, equalized_lr=equalized_lr)(feature_vector)

        self.model = kr.Model(input_image, [adv_value, latent_vector, latent_logvar, feature_vector])

    def call(self, inputs, *args, **kwargs):
        return self.model(inputs, *args, **kwargs)