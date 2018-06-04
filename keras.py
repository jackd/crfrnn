from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from .layers import CrfRnnLayer as BaseCrfRnnLayer


class CrfRnnLayer(BaseCrfRnnLayer, tf.keras.layers.Layer):
    pass
