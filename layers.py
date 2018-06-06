"""
MIT License

Copyright (c) 2017 Sadeep Jayasumana

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

Refactored by Dominic Jack
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

from .ops import lattice_filter
# from .ops import high_dim_filter
from .ops import fixed_point_iteration
from .ops import channels_first_to_channels_last
from .ops import channels_last_to_channels_first


def _diagonal_initializer(shape, dtype=tf.float32, **kwargs):
    if not len(shape) == 2 and shape[0] == shape[1]:
        raise ValueError('shape must be [n, n], got %s' % str(shape))
    return tf.eye(shape[0], dtype=tf.float32)


def _potts_model_initializer(shape, dtype=tf.float32, **kwargs):
    return -1 * _diagonal_initializer(shape, dtype, **kwargs)


_valid_data_formats = {'channels_first', 'channels_last'}


class CrfRnnLayerMixin(object):
    """
    Mixin class for CRF-RNN layer described in:

    Conditional Random Fields as Recurrent Neural Networks,
    S. Zheng, S. Jayasumana, B. Romera-Paredes, V. Vineet, Z. Su, D. Du,
    C. Huang and P. Torr, ICCV 2015

    See CrfRnnLayer and keras_layers.CrfRnnLayer for full implementations.
    """

    def __init__(self, theta_alpha, theta_beta, theta_gamma,
                 data_format='channels_last', fpi_kwargs={}, map_inputs=True,
                 map_kwargs={}):
        self.theta_alpha = theta_alpha
        self.theta_beta = theta_beta
        self.theta_gamma = theta_gamma
        self.spatial_ker_weights = None
        self.bilateral_ker_weights = None
        self.compatibility_matrix = None
        self.fpi_kwargs = fpi_kwargs
        self.map_inputs = map_inputs
        self.map_kwargs = map_kwargs
        self.data_format = data_format
        if data_format not in _valid_data_formats:
            raise ValueError(
                    'Invalid data_format %s. Must be in %s'
                    % (data_format, str(_valid_data_formats)))

    def _get_dims(self, input_shape):
        unaries_shape, rgb_shape = (tf.TensorShape(s) for s in input_shape)
        image_dims = tuple(rgb_shape.as_list()[1:3])
        num_classes = unaries_shape[-1].value
        return num_classes, image_dims

    def _build(self, input_shape):
        # Weights of the spatial kernel
        self.spatial_ker_weights = self.add_variable(
            name='spatial_ker_weights',
            shape=(self.num_classes, self.num_classes),
            initializer=_diagonal_initializer, trainable=True)

        # Weights of the bilateral kernel
        self.bilateral_ker_weights = self.add_variable(
            name='bilateral_ker_weights',
            shape=(self.num_classes, self.num_classes),
            initializer=_diagonal_initializer,
            trainable=True)

        # Compatibility matrix
        self.compatibility_matrix = self.add_variable(
            name='compatibility_matrix',
            shape=(self.num_classes, self.num_classes),
            initializer=_potts_model_initializer,
            trainable=True)

        self.all_ones = tf.ones(
            (self.num_classes,) + self.image_dims, dtype=np.float32)

    def get_norm_vals(self, softmax_out, rgb):
        spatial_norm_vals = lattice_filter(
            softmax_out, rgb, bilateral=False,
            theta_gamma=self.theta_gamma)
        bilateral_norm_vals = lattice_filter(
            softmax_out, rgb, bilateral=True, theta_alpha=self.theta_alpha,
            theta_beta=self.theta_beta)
        return spatial_norm_vals, bilateral_norm_vals

    def _step(self, q_values, unaries, rgb, spatial_norm_vals,
              bilateral_norm_vals):
        c = self.num_classes
        softmax_out = tf.nn.softmax(q_values, axis=1)

        # Spatial and Bilateral filtering
        spatial_out, bilateral_out = self.get_norm_vals(softmax_out, rgb)

        spatial_out = spatial_out / spatial_norm_vals
        bilateral_out = bilateral_out / bilateral_norm_vals

        # Weighting filter outputs
        message_passing = tf.matmul(
                self.spatial_ker_weights,
                tf.reshape(spatial_out, (-1, c)),
                transpose_b=True) + \
            tf.matmul(
                self.bilateral_ker_weights,
                tf.reshape(bilateral_out, (-1, c)),
                transpose_b=True)

        # Compatibility transform
        pairwise = tf.matmul(self.compatibility_matrix, message_passing)

        # Adding unary potentials
        pairwise = tf.reshape(pairwise, unaries.shape)
        q_values = unaries - pairwise
        return q_values

    def get_update_fn(self, unaries, rgb):
        spatial_norm_vals, bilateral_norm_vals = self.get_norm_vals(
            tf.ones_like(unaries), rgb)

        def f(q_values):
            return self._step(
                q_values, unaries, rgb, spatial_norm_vals,
                bilateral_norm_vals)

        return f

    def _get_q_values(self, inputs):
        unaries, rgb = inputs
        update_fn = self.get_update_fn(unaries, rgb)
        q_values = fixed_point_iteration(
            update_fn, (unaries,), **self.fpi_kwargs)

        return q_values

    def _call(self, inputs):
        if self.data_format == 'channels_first':
            inputs = [channels_first_to_channels_last(inp) for inp in inputs]
            q_values = self._get_q_values(inputs)
            return channels_last_to_channels_first(q_values)
        else:
            q_values = self._get_q_values(inputs)
            return q_values

    def compute_output_shape(self, input_shape):
        return input_shape


class CrfRnnLayer(tf.layers.Layer, CrfRnnLayerMixin):
    def __init__(self, theta_alpha=160.0, theta_beta=3.0, theta_gamma=3.0,
                 data_format='channels_last', fpi_kwargs={}, map_inputs=True,
                 map_kwargs={}, **kwargs):
        CrfRnnLayerMixin.__init__(
                self, theta_alpha, theta_beta, theta_gamma,
                data_format=data_format, fpi_kwargs=fpi_kwargs,
                map_inputs=map_inputs, map_kwargs=map_kwargs)
        super(CrfRnnLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        self.num_classes, self.image_dims = self._get_dims(input_shape)
        self._build(input_shape)
        super(CrfRnnLayer, self).build(input_shape)

    def call(self, inputs):
        return self._call(inputs)
