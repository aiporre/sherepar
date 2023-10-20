import unittest

from matplotlib import pyplot as plt
import numpy as np
from spherepar.visualization import render_volume, plot_slices_vol
# from skimage.draw import ellipsoid
import h5py as h5

class TestVisualization(unittest.TestCase):
    def test_vol_render(self):
        # basic_form = ellipsoid(10, 10, 10, levelset=False, spacing=(0.5, 0.5, 0.5))
        # make a cube and render it
        basic_form = np.zeros((10, 10, 10))
        basic_form[2:8, 2:8, 2:8] = 1.0
        # render the volume
        fig, ax = plt.subplots()
        render_volume(basic_form, azimuth=0, elevation=0, image_size=20, ax=ax)
        plt.show()
    def test_vol_reading_datafile(self):
        data = h5.File('../data/datacube.hdf5', 'r')
        datacube = np.array(data['density'])
        print(datacube.shape)
        # fig, ax = plt.subplots(1,3)
        # plot_slices_vol(datacube, ax=ax)
        # plt.show()

        fig, ax = plt.subplots()
        render_volume(datacube, azimuth=0, elevation=0, image_size=180, ax=ax)
        plt.show()


