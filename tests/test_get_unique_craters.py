from __future__ import absolute_import, division, print_function
import numpy as np
import h5py
import pandas as pd
import sys
sys.path.append('../')
import deepmars.utils.transform as guc


class TestLongLatEstimation(object):

    def setup(self):
        ctrs = pd.HDFStore('./ran_craters_175000.hdf5', 'r')
        ctrs_meta = h5py.File('./ran_images_175000.hdf5', 'r')
        img = 'img_175001'
        self.craters = ctrs[img]
        self.dim = (256, 256)
        self.llbd = ctrs_meta['longlat_bounds'][img][...]
        self.dc = ctrs_meta['pix_distortion_coefficient'][img][...]
        ctrs.close()
        ctrs_meta.close()

    def test_estimate_longlatdiamkm(self):
        coords = self.craters[['x', 'y', 'Diameter (pix)']].values
        craters_unique = guc.estimate_longlatdiamkm(
            self.dim, self.llbd, self.dc, coords)
        # Check that estimate is same as predictions in sample_crater_csv.hdf5.
        assert np.all(np.isclose(craters_unique[:, 0],
                                 self.craters['Long'],
                                 atol=0., rtol=1e-2))
        assert np.all(np.isclose(craters_unique[:, 1],
                                 self.craters['Lat'],
                                 atol=0., rtol=1e-2))
        assert np.all(np.isclose(craters_unique[:, 2],
                                 self.craters['Diameter (km)'],
                                 atol=0., rtol=1e-2))
        # Check that estimate is within expected tolerance from ground truth
        # values in sample_crater_csv.hdf5.
        assert np.all(abs(craters_unique[:, 0] - self.craters['Long']) /
                      (self.llbd[1] - self.llbd[0]) < 0.01)
        assert np.all(abs(craters_unique[:, 1] - self.craters['Lat']) /
                      (self.llbd[3] - self.llbd[2]) < 0.02)
        # Radius is exact, since we use the inverse estimation from km to pix
        # to get the ground truth crater pixel radii/diameters in
        # input_data_gen.py.
        assert np.all(np.isclose(craters_unique[:, 2],
                                 self.craters['Diameter (km)'],
                                 atol=0., rtol=1e-10))
