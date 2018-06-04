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

from .high_dim_filter import high_dim_filter


def _diagonal_initializer(shape, dtype=tf.float32, **kwargs):
    if not len(shape) == 2 and shape[0] == shape[1]:
        raise ValueError('shape must be [n, n], got %s' % str(shape))
    return tf.eye(shape[0], dtype=tf.float32)


def _potts_model_initializer(shape, dtype=tf.float32, **kwargs):
    return -1 * _diagonal_initializer(shape, dtype, **kwargs)


def _channels_last_to_channels_first(x):
    return tf.transpose(x, (0, 3, 1, 2))


def _channels_first_to_channels_last(x):
    return tf.transpose(x, (0, 2, 3, 1))

_valid_data_formats = {'channels_first', 'channels_last'}


class CrfRnnLayerMixin(object):
    """
    Implements the CRF-RNN layer described in:

    Conditional Random Fields as Recurrent Neural Networks,
    S. Zheng, S. Jayasumana, B. Romera-Paredes, V. Vineet, Z. Su, D. Du,
    C. Huang and P. Torr, ICCV 2015
    """

    def __init__(self, image_dims, num_classes,
                 theta_alpha, theta_beta, theta_gamma,
                 num_iterations,
                 data_format='channels_last',
                 explicit_loop=True, loop_kwargs={},
                 map_inputs=True, map_kwargs={}):
        self.image_dims = tuple(image_dims)
        self.num_classes = num_classes
        self.theta_alpha = theta_alpha
        self.theta_beta = theta_beta
        self.theta_gamma = theta_gamma
        self.num_iterations = num_iterations
        self.spatial_ker_weights = None
        self.bilateral_ker_weights = None
        self.compatibility_matrix = None
        self.explicit_loop = explicit_loop
        self.loop_kwargs = loop_kwargs
        self.map_inputs = map_inputs
        self.map_kwargs = map_kwargs
        self.data_format = data_format
        if data_format not in _valid_data_formats:
            raise ValueError(
                    'Invalid data_format %s. Must be in %s'
                    % (data_format, str(_valid_data_formats)))
        # super(CrfRnnLayer, self).__init__(**kwargs)

    def _build(self, input_shape):
        # Weights of the spatial kernel
        self.spatial_ker_weights = self.add_weight(
            name='spatial_ker_weights',
            shape=(self.num_classes, self.num_classes),
            initializer=_diagonal_initializer, trainable=True)

        # Weights of the bilateral kernel
        self.bilateral_ker_weights = self.add_weight(
            name='bilateral_ker_weights',
            shape=(self.num_classes, self.num_classes),
            initializer=_diagonal_initializer,
            trainable=True)

        # Compatibility matrix
        self.compatibility_matrix = self.add_weight(
            name='compatibility_matrix',
            shape=(self.num_classes, self.num_classes),
            initializer=_potts_model_initializer,
            trainable=True)

        self.all_ones = tf.ones(
            (self.num_classes,) + self.image_dims, dtype=np.float32)

        # super(CrfRnnLayer, self).build(input_shape)

    def _get_norm_vals(self, softmax_out, rgb):
        spatial_norm_vals = high_dim_filter(
            softmax_out, rgb, bilateral=False,
            theta_gamma=self.theta_gamma)
        bilateral_norm_vals = high_dim_filter(
            softmax_out, rgb, bilateral=True, theta_alpha=self.theta_alpha,
            theta_beta=self.theta_beta)
        return spatial_norm_vals, bilateral_norm_vals

    def _get_init_norm_vals(self, rgb):
        return self._get_norm_vals(self.all_ones, rgb)

    def get_norm_vals(self, softmax_out, rgb):
        if softmax_out is None:
            def fn(rgb):
                return self._get_norm_vals(self.all_ones, rgb)
            inputs = rgb
        else:
            def fn(inputs):
                return self._get_norm_vals(*inputs)
            inputs = softmax_out, rgb
        return tf.map_fn(
            fn, inputs, (tf.float32, tf.float32),
            **self.map_kwargs)

    def _step(self, q_values, unaries, rgb, spatial_norm_vals,
              bilateral_norm_vals):
        c = self.num_classes
        softmax_out = tf.nn.softmax(q_values, axis=1)

        # Spatial and Bilateral filtering
        spatial_out, bilateral_out = self.get_norm_vals(softmax_out, rgb)

        spatial_out = spatial_out / spatial_norm_vals
        bilateral_out = bilateral_out / bilateral_norm_vals

        hw = self.image_dims[0] * self.image_dims[1]
        # Weighting filter outputs

        def batch_matmul(w, x):
            return tf.einsum('ij,kjl->kil', w, x)

        message_passing = (batch_matmul(
            self.spatial_ker_weights, tf.reshape(spatial_out, (-1, c, hw))) +
            batch_matmul(
                self.bilateral_ker_weights,
                tf.reshape(bilateral_out, (-1, c, hw))))

        # Compatibility transform
        pairwise = batch_matmul(self.compatibility_matrix, message_passing)

        # Adding unary potentials
        pairwise = tf.reshape(pairwise, (-1, c) + self.image_dims)
        q_values = unaries - pairwise
        return q_values

    def _get_q_values(self, inputs):
        unaries, rgb = inputs
        spatial_norm_vals, bilateral_norm_vals = self.get_norm_vals(None, rgb)

        if self.explicit_loop:
            q_values = unaries
            for i in range(self.num_iterations):
                q_values = self._step(
                    q_values, unaries, rgb, spatial_norm_vals,
                    bilateral_norm_vals)
        else:
            def cond(*args):
                return True

            def body(q_values):
                return self._step(
                    q_values, unaries, rgb, spatial_norm_vals,
                    bilateral_norm_vals)

            q_values = tf.while_loop(
                cond, body, (unaries,),
                maximum_iterations=self.num_iterations, **self.loop_kwargs)

        return q_values

    def _call(self, inputs):
        if self.data_format == 'channels_last':
            inputs = [_channels_last_to_channels_first(inp) for inp in inputs]
            q_values = self._get_q_values(inputs)
            return _channels_first_to_channels_last(q_values)
        else:
            q_values = self._get_q_values(inputs)
            return q_values

    def compute_output_shape(self, input_shape):
        return input_shape


class CrfRnnLayer(tf.layers.Layer, CrfRnnLayerMixin):
    def __init__(self, image_dims, num_classes,
                 theta_alpha, theta_beta, theta_gamma,
                 num_iterations,
                 data_format='channels_last',
                 explicit_loop=True, loop_kwargs={},
                 map_inputs=True, map_kwargs={}, **kwargs):
        CrfRnnLayerMixin.__init__(
                self, image_dims, num_classes,
                theta_alpha, theta_beta, theta_gamma,
                num_iterations,
                data_format=data_format,
                explicit_loop=explicit_loop, loop_kwargs=loop_kwargs,
                map_inputs=map_inputs, map_kwargs=map_kwargs)
        super(CrfRnnLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        self._build(input_shape)
        super(CrfRnnLayer, self).build(input_shape)

    def call(self, inputs):
        return self._call(inputs)