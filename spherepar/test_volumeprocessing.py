import unittest
import numpy as np
from skimage.draw import ellipsoid
from spherepar.volumeprocessing import interpolate_volume
import matplotlib.pyplot as plt
# import
from unittest.mock import MagicMock



class TestMeshSurf(unittest.TestCase):
    def test_interpolate_volume(self):
        # test the interpolation of a sphere
        basic_form = ellipsoid(10, 10, 10, levelset=False, spacing=(0.5, 0.5, 0.5))
        # interpolate the volume and measure the volume
        factor = 2
        interpolated_volume = interpolate_volume(basic_form, factor, binary=False)
        new_shape = tuple(np.array(basic_form.shape) * np.array(factor))
        self.assertEqual(interpolated_volume.shape, new_shape)
        self.assertAlmostEqual(interpolated_volume.sum()/factor**3,basic_form.sum()-0.75)
        # test the interpolation of a box
        basic_form = np.zeros((10, 10, 10))
        basic_form[2:8, 2:8, 2:8] = 1.0
        # interpolate the volume and measure the volume
        interpolated_volume = interpolate_volume(basic_form, factor)
        new_shape = tuple(np.array(basic_form.shape) * np.array(factor))
        self.assertEqual(interpolated_volume.shape, new_shape)
        self.assertAlmostEqual(interpolated_volume.sum()/factor**3, basic_form.sum())

    def test_plot_is_generated(self):
        image_mock = np.array([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.25, 0.5, 0.5, 0.5, 0.5, 0.5, 0.25, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.5, 1.0, 1.0, 1.0, 1.0, 1.0, 0.5, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.5, 1.0, 1.0, 1.0, 1.0, 1.0, 0.5, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.5, 1.0, 1.0, 1.0, 1.0, 1.0, 0.5, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.5, 1.0, 1.0, 1.0, 1.0, 1.0, 0.5, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.5, 1.0, 1.0, 1.0, 1.0, 1.0, 0.5, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.25, 0.5, 0.5, 0.5, 0.5, 0.5, 0.25, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]])
        fig, ax = plt.subplots()
        basic_form = ellipsoid(1, 1, 1, levelset=False, spacing=(0.5, 0.5, 0.5))
        # interpolate the volume and measure the volume
        factor = 2
        interpolated_volume = interpolate_volume(basic_form, factor, binary=False, ax=ax)
        plt.show()
        axes_image = ax.get_images()[0]
        print(axes_image)
        data = axes_image.get_array()
        print(data.shape)
        self.assertTrue((data == image_mock).all())


     
