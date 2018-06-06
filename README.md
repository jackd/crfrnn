Tensorflow implementation of CrfRnn layer according to [Conditional Random Fields as Recurrent Neural Networks](http://www.robots.ox.ac.uk/~szheng/papers/CRFasRNN.pdf).

This is a minor repackaging of [this repo](https://github.com/MiguelMonteiro/CRFasRNNLayer), which itself is based on the work of [this repo](https://github.com/sadeepj/crfasrnn_keras).

Note this repo does *not* contain the kernel for the lattice dimensional filter. See [Setup](#setup) below.

### Setup
1. Clone this repository and add the parent directory to your `PYTHONPATH`
```
cd /path/to/parent_dir
git clone https://github.com/jackd/crfrnn
export PYTHONPATH=$PYTHONPATH:/path/to/parent_dir
```
2. Get the source code for the kernel.
```
git clone --recursive https://github.com/MiguelMonteiro/CRFasRNNLayer
```
3. change the number of channels to an appropriate value (to run the examples, this must be 21).
4. build - you may need to [manually install the latest version of cmake](https://askubuntu.com/questions/355565/how-do-i-install-the-latest-version-of-cmake-from-the-command-line). Note you'll need to change `INPUT_CHANNELS` to the number of classes.
5. Copy `lattice_filter.so` to `./lattice_filter/lattice_filter.so`

### Keras Demo
To run the demo:
1. Follow [Setup](#setup) instructions above.
2. Download pretrained model weights [here](https://goo.gl/ciEYZi).
3. Move weights to `./example/keras/crfrnn_keras_model.h5`.
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
