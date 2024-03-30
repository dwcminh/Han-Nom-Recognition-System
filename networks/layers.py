import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Conv2DTranspose, BatchNormalization, ReLU


class ConvBnRelu(tf.keras.layers.Layer):
    def __init__(self, filters=64, kernel_size=3, padding='same', use_bias=True, name='ConvBnRelu', **kwargs):
        super(ConvBnRelu, self).__init__(name=name, **kwargs)
        self.conv = Conv2D(filters, kernel_size, padding=padding, use_bias=use_bias)
        self.bn = BatchNormalization()
        self.relu = ReLU()
        self.filters = filters
        self.kernel_size = kernel_size
        self.padding = padding
        self.use_bias = use_bias
        
    def call(self, inputs, training):
        x = self.conv(inputs)
        x = self.bn(x, training=training)
        return self.relu(x)
    
    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'filters': self.filters,
            'kernel_size': self.kernel_size,
            'padding': self.padding,
            'use_bias': self.use_bias
        })
        return config


class DeConvMap(tf.keras.layers.Layer):
    def __init__(self, filters=64, name='DeConvMap', **kwargs):
        super(DeConvMap, self).__init__(name=name, **kwargs)
        self.conv_bn = ConvBnRelu(filters, kernel_size=3, use_bias=False)
        self.deconv1 = Conv2DTranspose(filters, kernel_size=2, strides=2, use_bias=False)
        self.bn = BatchNormalization()
        self.relu = ReLU()
        self.deconv2 = Conv2DTranspose(1, kernel_size=2, strides=2, activation='sigmoid')
        self.filters = filters
        
    def call(self, inputs, training):
        x = self.conv_bn(inputs)
        x = self.deconv1(x)
        x = self.bn(x, training=training)
        x = self.relu(x)
        x = self.deconv2(x)
        return tf.squeeze(x, axis=-1)
    
    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'filters': self.filters
        })
        return config
    
class ApproximateBinaryMap(tf.keras.layers.Layer):
    def __init__(self, k, **kwargs):
        super(ApproximateBinaryMap, self).__init__(**kwargs)
        self.k = k

    def call(self, inputs):
        # Unpack the inputs. Expecting inputs to be in the form [P, T]
        P, T = inputs
        # Perform the computation 1 / (1 + exp(-k * (P - T)))
        return 1 / (1 + tf.exp(-self.k * (P - T)))

    def get_config(self):
        config = super(ApproximateBinaryMap, self).get_config()
        config.update({'k': self.k})
        return config