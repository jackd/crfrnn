from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import keras
from .layers import CrfRnnLayerMixin


class CrfRnnLayer(keras.engine.topology.Layer, CrfRnnLayerMixin):
    def __init__(self, image_dims, num_classes,
                 theta_alpha, theta_beta, theta_gamma,
                 num_iterations,
                 data_format='channels_last',
                 explicit_loop=True, loop_kwargs={},
                 map_inputs=True, map_kwargs={}, **kwargs):

        super(CrfRnnLayer, self).__init__(**kwargs)
        CrfRnnLayerMixin.__init__(
                self, image_dims, num_classes,
                theta_alpha, theta_beta, theta_gamma,
                num_iterations,
                data_format=data_format,
                explicit_loop=explicit_loop, loop_kwargs=loop_kwargs,
                map_inputs=map_inputs, map_kwargs=map_kwargs)

        # super(CrfRnnLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        self._build(input_shape)
        super(CrfRnnLayer, self).build(input_shape)

    def call(self, inputs):
        return self._call(inputs)

