# DeepMars - Mars Crater Counting Through Deep Learning

DeepMars is a TensorFLow based pipeline for training a UNET to recognize craters
on Mars, and determine their location and radius using a circle matching algorithm.

## Getting Started

### Overview

The Crater Detection Algorithm uses a trained neural network to detect ring structures from Digital Elevation Models (DEM) of Mars. Once these ring structures are found a Fourier based template matching algorithm identifies the circles and determines their position and size based on the image parameters. The code to train the neural network is also included - In the included code about 30,000 images of craters on Mars are used, along with the Robbins and Hynek (2012) crater list to train the CNN to identify craters as ring structures.

### Dependencies

DeepMars requires the follows

- [Python](https://www.python.org/) version 2.7 or 3.5+, tested on 3.6
- [Cartopy](http://scitools.org.uk/cartopy/) >= 0.14.2.  Cartopy itself has a
number of [dependencies](http://scitools.org.uk/cartopy/docs/latest/installing.html#installing),
including the GEOS and Proj.4.x libraries.  (For Ubuntu systems, these can be
installed through the `libgeos++-dev` and `libproj-dev` packages,
respectively.)
- [h5py](http://www.h5py.org/) >= 2.6.0
- [Keras](https://keras.io/) 1.2.2 [(documentation)](https://faroit.github.io/keras-docs/1.2.2/);
  also tested with Keras >= 2.0.2
- [Numpy](http://www.numpy.org/) >= 1.12
- [OpenCV](https://pypi.python.org/pypi/opencv-python) >= 3.2.0.6
- [Pandas](https://pandas.pydata.org/) >= 0.19.1
- [Pillow](https://python-pillow.org/) >= 3.1.2
- [PyTables](http://www.pytables.org/) >=3.4.2
- [TensorFlow](https://www.tensorflow.org/) 0.10.0rc0, also tested with
  TensorFlow >= 1.0

### Data Sources

#### Digital Elevation Models

We use the [MOLA+HRSC combined DEM][molahrsc]. The DEM is created from two partial DEMs of Mars and is intended to have 200m/pixel resolution. Approximately 45% of the DEM is at this resolution or higher, and the remainder is at the 400m/pixel resolution of MOLA. The source DEM at 16bit/pixel is used by the algorithm to retain the full topographic resolution. 


#### Crater Catalogue

We use the [Robbins and Hynek (2012)][robbins] crater catalogue to provide the ground-truth / training data used in the CNN and for comparison with all of the craters found by the CDA. The catalogue is available as a comma separated file (CSV). 

#### Running DeepMars

Each stage of the DeepMars pipeline has an executable script. `deepmars/data/make_dataset.py` creates the crater dataset by sampling the input DEM and crater catalogue and outputing randomly or systematically sampled craters in HDF5 files. `deepmars/models/train_model_sys.py` trains the CNN using the created dataset, using either ring based algorithm and binary cross entropy or a disk based algorthim with intersection-over-union metrics. `deepmars/models/predict_model.py` makes predictions from images using the CNN and template matching as needed, individual parts of the algorithm can be run on the whole dataset or single images for testing.

## Authors

DeepMars is based on the DeepMoon CDA developed by Silburt et al [2019]. The core code is largely un-modified from DeepMoon. The main differences here are the the structure of the code, with clearer separation of functions in to files and adding arguments to executable code, the use of a 16 bit image inside Python instead of converting to 8-bit PNGs externally, and the addition of a disk finding algorithm and intersection-over-union metric.

**Christopher Lee** - Modifications and re-formatting for DeepMars [eelsirhc](https://github.com/eelsirhc)

DeepMoon authours - [**Ari Silburt**](https://github.com/silburt). [**Charles Zhu**](https://github.com/cczhu)


## License

Copyright 2018-2019 Christopher Lee, Ari Silburt, Charles Zhu and contributors.

[molahrsc]: https://astrogeology.usgs.gov/search/map/Mars/Topography/HRSC_MOLA_Blend/Mars_HRSC_MOLA_BlendDEM_Global_200mp_v2
[robbins]: https://astrogeology.usgs.gov/search/map/Mars/Research/Craters/RobbinsCraterDatabase_20120821
