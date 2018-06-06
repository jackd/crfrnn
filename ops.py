from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
# from .high_dim_filter import high_dim_filter  # NOQA
from .lattice_filter import module as _module

lattice_filter = _module.lattice_filter


def unrolled_crf_rnn(unaries, rgb, **kwargs):
    from .layers import CrfRnnLayer
    layer = CrfRnnLayer(**kwargs)
    return layer((unaries, rgb))


def fixed_point_iteration(
        f, x0, num_iterations=10, explicit_loop=True, **loop_kwargs):
    if explicit_loop:
        x = x0
        for i in range(num_iterations):
            if isinstance(x, tf.Tensor):
                x = x,
            x = f(*x)
    else:
        def cond(*args):
            return True

        if isinstance(x0, tf.Tensor):
            x0 = x0,
        x = tf.while_loop(cond, f, x0, maximum_iterations=num_iterations,
                          **loop_kwargs)
    return x


def channels_last_to_channels_first(x):
    return tf.transpose(x, (0, 3, 1, 2))


def channels_first_to_channels_last(x):
    return tf.transpose(x, (0, 2, 3, 1))
