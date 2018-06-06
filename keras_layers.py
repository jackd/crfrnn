from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# import keras
from keras.engine.topology import Layer
# import tensorflow as tf
from .layers import CrfRnnLayerMixin
# Layer = tf.keras.layers.Layer


class CrfRnnLayer(Layer, CrfRnnLayerMixin):
    def __init__(self, image_dims, num_classes,
                 theta_alpha=160.0, theta_beta=3.0, theta_gamma=3.0,
                 data_format='channels_last',
                 fpi_kwargs={}, map_inputs=True, map_kwargs={}, **kwargs):
        self.image_dims = tuple(image_dims)
        self.num_classes = num_classes
        CrfRnnLayerMixin.__init__(
            self, theta_alpha, theta_beta, theta_gamma,
            data_format=data_format, fpi_kwargs=fpi_kwargs,
            map_inputs=map_inputs, map_kwargs=map_kwargs)
        super(CrfRnnLayer, self).__init__(**kwargs)

    def add_variable(self, *args, **kwargs):
        return self.add_weight(*args, **kwargs)

    def build(self, input_shape):
        num_classes, image_dims = self._get_dims(input_shape)
        if image_dims != self.image_dims:
            raise ValueError(
                'image_dims inconsistent with value provided in constructor')
        if num_classes != self.num_classes:
            raise ValueError(
                'num_classes inconsistent with value provided in constructor')
        self._build(input_shape)
        super(CrfRnnLayer, self).build(input_shape)

    def call(self, inputs):
        return self._call(inputs)
