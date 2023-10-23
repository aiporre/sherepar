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
        basic_form = np.zeros((100, 100, 100))
        basic_form[20:80, 20:80, 20:80] = 1.0
        # render the volume
        fig, ax = plt.subplots()
        render_volume(basic_form, azimuth=45, elevation=45, image_size=100, ax=ax,
                      values=[0.5], colors=['#ffff24'], alphas=[.9])
        plt.show()
    def test_vol_reading_datafile(self):
        data = h5.File('../data/datacube.hdf5', 'r')
        datacube = np.array(data['density'])
        print(datacube.shape)
        # fig, ax = plt.subplots(1,3)
        # plot_slices_vol(datacube, ax=ax)
        # plt.show()

        fig, ax = plt.subplots()
        render_volume(datacube, azimuth=0, elevation=0, image_size=180,
                      ax=ax, values=[9,3,-3],
                      colors=['#ffff24', '#24ff24', '#2424ff'],
                      alphas=[0.6, 0.1, 0.01])
        plt.show()


