"""Example usage of native tensorflow layer."""
import tensorflow as tf

batch_size = 2
image_dims = (224, 224)
n_classes = 5


def _run(out_fn):
    graph = tf.Graph()
    with graph.as_default():
        image = tf.random_normal(
            shape=(batch_size,) + image_dims + (3,), dtype=tf.float32)
        logits = tf.random_normal(
            shape=(batch_size,) + image_dims + (n_classes,), dtype=tf.float32)

        out = out_fn(logits, image)
        init = tf.global_variables_initializer()

    with tf.Session(graph=graph) as sess:
        sess.run(init)
        sess.run(out)


def run_layer():
    from crfrnn.layers import CrfRnnLayer

    def f(logits, image):
        layer = CrfRnnLayer()
        return layer((logits, image))
    _run(f)
    print('Done layer!')


def run_op():
    from crfrnn.ops import unrolled_crf_rnn
    _run(unrolled_crf_rnn)
    print('Done op!')


run_layer()
run_op()
