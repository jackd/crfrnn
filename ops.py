from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from .high_dim_filter import high_dim_filter  # NOQA


def unrolled_crf_rnn(unaries, rgb, **kwargs):
    from .layers import CrfRnnLayer
    layer = CrfRnnLayer(**kwargs)
    return layer((unaries, rgb))
