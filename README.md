Tensorflow implementation of CrfRnn layer according to [Conditional Random Fields as Recurrent Neural Networks](http://www.robots.ox.ac.uk/~szheng/papers/CRFasRNN.pdf).

This is a minor repackaging of [this repository](https://github.com/sadeepj/crfasrnn_keras) with the following changes:
* repackaged for importing into other projects
* keras-independent implementation (plus a keras wrapper implementation)
* `tf.map_fn` allows for `batch_size > 1`
* `tf.while_loop` implementation allows flexibility
* `channels_first` or `channels_last` data format

Note this repo does *not* contain the kernel for the high dimensional filter. See [Setup](#setup) below.

### Setup
1. Clone this repository and add the parent directory to your `PYTHONPATH`
```
cd /path/to/parent_dir
git clone https://github.com/jackd/crfrnn
export PYTHONPATH=$PYTHONPATH:/path/to/parent_dir
```
2. Follow instructions [here](https://github.com/sadeepj/crfasrnn_keras) to build the kernel.
3. Copy the built `.so` file to `./high_dim_filter`
```
cd /path/to/parent_dir/crfrnn
cp /path/to/built/high_dim_filter.so ./high_dim_filter/
```

### Keras Demo
To run the demo:
1. Follow [Setup](#setup) instructions above.
2. Download pretrained model weights [here](https://goo.gl/ciEYZi).
3. Move weights to `example/keras/crfrnn_keras_model.h5`.
4. Run the `run_demo.py` script.
```
cd /path/to/parent_dir/crfrnn/example/keras
mv ~/Downloads/crfrnn_keras_model.h5 ./crfrnn_keras_model.h5
./run_demo.py
```

### Acknowledgement
As per the original repo, if you use this code/model for your research, please cite the following paper:
```
@inproceedings{crfasrnn_ICCV2015,
    author = {Shuai Zheng and Sadeep Jayasumana and Bernardino Romera-Paredes and Vibhav Vineet and
    Zhizhong Su and Dalong Du and Chang Huang and Philip H. S. Torr},
    title  = {Conditional Random Fields as Recurrent Neural Networks},
    booktitle = {International Conference on Computer Vision (ICCV)},
    year   = {2015}
}
```
