#!/usr/bin/python
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
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import numpy as np
from .util import get_preprocessed_image, get_label_image
from .crfrnn_model import get_crfrnn_model_def


def main(save=True):
    folder = os.path.dirname(__file__)
    input_file = os.path.join('image.jpg', folder)

    # Download the model from https://goo.gl/ciEYZi
    saved_model_path = os.path.join(folder, 'crfrnn_keras_model.h5')
    if not os.path.isfile(saved_model_path):
        raise IOError('No file at %s. Download from https://goo.gl/ciEYZi'
                      % saved_model_path)

    img_data, img_h, img_w = get_preprocessed_image(input_file)
    img_data = np.concatenate((img_data, img_data[:, :, -1::-1, :]), axis=0)

    model = get_crfrnn_model_def()
    model.load_weights(saved_model_path)
    probs = model.predict(img_data, verbose=False)

    if save:
        for i, prob in enumerate(probs):
            get_label_image(probs, img_h, img_w).save(
                os.path.join(folder, 'out%d.png' % i))

    else:
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, len(probs))
        if len(probs) == 1:
            axes = [axes]
        for ax, prob in zip(axes, probs):
            ax.imshow(get_label_image(prob, img_h, img_w))
        plt.show()


if __name__ == '__main__':
    main()
