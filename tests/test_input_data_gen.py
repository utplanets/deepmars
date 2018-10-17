from __future__ import absolute_import, division, print_function
import sys
import pytest
import pandas as pd
import numpy as np
import cv2
import h5py
import cartopy.crs as ccrs
import cartopy.img_transform as cimg
from PIL import Image
sys.path.append('../')
from deepmars.data.common import ReadRobbinsCraters

# import input_data_gen as igen
# import utils.transform as trf


class TestCatalogue(object):
    """Tests crater catalogues."""

    def setup(self):
        # Head et al. dataset.
        robbins = pd.read_csv('../data/raw/RobbinsCraters_20121016.tsv',
                              header=0, sep='\t', engine='python')

        keep_columns = ["LATITUDE_CIRCLE_IMAGE",
                        "LONGITUDE_CIRCLE_IMAGE",
                        "DIAM_CIRCLE_IMAGE"]

        robbins = robbins[keep_columns]
        robbins.columns = ["Lat", "Long", "Diameter (km)"]

        self.robbins_t = robbins
        self.robbins_t.sort_values(by='Lat', inplace=True)
        self.robbins_t.reset_index(inplace=True, drop=True)

    def test_dataframes_equal(self):
        robbins = ReadRobbinsCraters("../data/raw/RobbinsCraters_20121016.tsv")

        robbins.reset_index(inplace=True, drop=True)
        assert not np.all(robbins.values == self.robbins_t.values)
        assert np.all(robbins.sort_values("Lat").values ==
                      self.robbins_t.values)
