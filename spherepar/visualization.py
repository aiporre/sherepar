from typing import Any

import numpy as np
import matplotlib.pyplot as plt
from numpy import ndarray, dtype, floating, float_
from numpy._typing import _64Bit
from scipy.interpolate import interpn


def plot_slices_vol(volume: np.ndarray, ax=None):
    """
    Plot the x, y and z axis slices of a volume.
    :param volume:
    :type volume: np.ndarray
    """
    # get the x, y and z coordinates
    x, y, z = np.nonzero(volume)
    # get the minimum and maximum indices along each axis
    min_idx = np.min([x, y, z], axis=1)
    max_idx = np.max([x, y, z], axis=1)
    # crop the volume using the minimum and maximum indices
    volume = volume[min_idx[0]:max_idx[0] + 1, min_idx[1]:max_idx[1] + 1, min_idx[2]:max_idx[2] + 1]
    # plot x y and z axis slices
    if ax is None:
        fig, ax = plt.subplots(1, 3)
        fig.set_size_inches(20, 20)
    N = volume.shape[0] // 2
    if ax is None:
        fig, ax = plt.subplots(1, 3)
    ax[0].imshow(volume[N, :, :])
    N = volume.shape[1] // 2
    ax[1].imshow(volume[:, N, :])
    N = volume.shape[2] // 2
    ax[2].imshow(volume[:, :, N])
    return ax

def generate_transfer_function(values: tuple[float, ...], colors: tuple[str,...], alphas: tuple[float, ...]):
    # validate the max number values, colors and alphas is not greater than 3
    assert len(values) <= 3, 'The number of values must be less or equal than 3'
    assert len(colors) <= 3, 'The number of colors must be less or equal than 3'
    assert len(alphas) <= 3, 'The number of alphas must be less or equal than 3'
    colors = [colors.replace('#', '') for colors in colors]
    colors = [tuple(int(colors[i][j:j + 2], 16)/255.0 for j in (0, 2, 4)) for i in range(len(colors))]
    if len(values) == 1:
        c = colors[0]
        def transferFunction(x):
            r = c[0] * np.exp(-(x - values[0]) ** 2 / 1.0)
            g = c[1] * np.exp(-(x - values[0]) ** 2 / 1.0)
            b = c[2] * np.exp(-(x - values[0]) ** 2 / 1.0)
            a = alphas[0] * np.exp(-(x - values[0]) ** 2 / 1.0)
            return r, g, b, a
        return transferFunction
    elif len(values) == 2:
        c1, c2 = colors
        def transferFunction(x):
            r = c1[0] * np.exp(-(x - values[0]) ** 2 / 1.0) + c2[0] * np.exp(-(x - values[1]) ** 2 / 1.0)
            g = c1[1] * np.exp(-(x - values[0]) ** 2 / 1.0) + c2[1] * np.exp(-(x - values[1]) ** 2 / 1.0)
            b = c1[2] * np.exp(-(x - values[0]) ** 2 / 1.0) + c2[2] * np.exp(-(x - values[1]) ** 2 / 1.0)
            a = alphas[0] * np.exp(-(x - values[0]) ** 2 / 1.0) + alphas[1] * np.exp(-(x - values[1]) ** 2 / 1.0)
            return r, g, b, a
        return transferFunction
    else:
        # convert colors text hex into rgb
        c1, c2, c3 = colors
        def transferFunction(x):
            r = c1[0] * np.exp(-(x - values[0]) ** 2 / 1.0) + c2[0] * np.exp(-(x - values[1]) ** 2 / 1.0) + c3[0] * np.exp(-(x - values[2]) ** 2 / 1.0)
            g = c1[1] * np.exp(-(x - values[0]) ** 2 / 1.0) + c2[1] * np.exp(-(x - values[1]) ** 2 / 1.0) + c3[1] * np.exp(-(x - values[2]) ** 2 / 1.0)
            b = c1[2] * np.exp(-(x - values[0]) ** 2 / 1.0) + c2[2] * np.exp(-(x - values[1]) ** 2 / 1.0) + c3[2] * np.exp(-(x - values[2]) ** 2 / 1.0)
            a = alphas[0] * np.exp(-(x - values[0]) ** 2 / 1.0) + alphas[1] * np.exp(-(x - values[1]) ** 2 / 1.0) + alphas[2] * np.exp(-(x - values[2]) ** 2 / 1.0)
            return r, g, b, a
        return transferFunction
        # def transferFunction(x):
        #     r = 1.0 * np.exp(-(x - 9.0) ** 2 / 1.0) + 0.1 * np.exp(-(x - 3.0) ** 2 / 0.1) + 0.1 * np.exp(
        #         -(x - -3.0) ** 2 / 0.5)
        #     g = 1.0 * np.exp(-(x - 9.0) ** 2 / 1.0) + 1.0 * np.exp(-(x - 3.0) ** 2 / 0.1) + 0.1 * np.exp(
        #         -(x - -3.0) ** 2 / 0.5)
        #     b = 0.1 * np.exp(-(x - 9.0) ** 2 / 1.0) + 0.1 * np.exp(-(x - 3.0) ** 2 / 0.1) + 1.0 * np.exp(
        #         -(x - -3.0) ** 2 / 0.5)
        #     a = 0.6 * np.exp(-(x - 9.0) ** 2 / 1.0) + 0.1 * np.exp(-(x - 3.0) ** 2 / 0.1) + 0.01 * np.exp(
        #         -(x - -3.0) ** 2 / 0.5)
        #     return r, g, b, a
        return transferFunction

def render_volume(volume: np.ndarray, azimuth: float = 0, elevation: float = 0, image_size: int = 100,
                  ax: plt.Axes = None, values: list = [0, 1, 2],
                  colors: list = ['#ff0000', '#00ff00', '#0000ff'], alphas: list = [0.1, 0.1, 0.1], degrees=True):
    """
    Render a volume as a 2D image.

    :param volume: data to render
    :param azimuth:  azimuth angle of the camera
    :param elevation: elevation angle of the camera
    :param image_size: image size will be image_size x image_size
    :return:
        Nothing, but shows the rendered image
    """

    # def transferFunction(x):
    #     r = 1.0 * np.exp(-(x - 9.0) ** 2 / 1.0) + 0.1 * np.exp(-(x - 3.0) ** 2 / 0.1) + 0.1 * np.exp(
    #         -(x - -3.0) ** 2 / 0.5)
    #     g = 1.0 * np.exp(-(x - 9.0) ** 2 / 1.0) + 1.0 * np.exp(-(x - 3.0) ** 2 / 0.1) + 0.1 * np.exp(
    #         -(x - -3.0) ** 2 / 0.5)
    #     b = 0.1 * np.exp(-(x - 9.0) ** 2 / 1.0) + 0.1 * np.exp(-(x - 3.0) ** 2 / 0.1) + 1.0 * np.exp(
    #         -(x - -3.0) ** 2 / 0.5)
    #     a = 0.6 * np.exp(-(x - 9.0) ** 2 / 1.0) + 0.1 * np.exp(-(x - 3.0) ** 2 / 0.1) + 0.01 * np.exp(
    #         -(x - -3.0) ** 2 / 0.5)
    #     return r, g, b, a

    # convert degrees to radians
    if degrees:
        azimuth = np.deg2rad(azimuth)
        elevation = np.deg2rad(elevation)

    # get the x, y and z coordinates centered middle of the volume
    Nx, Ny, Nz = volume.shape
    x, y, z = np.linspace(-Nx / 2, Nx / 2, Nx), np.linspace(-Ny / 2, Ny / 2, Ny), np.linspace(-Nz / 2, Nz / 2, Nz)
    # volument points are the coordinates of the volume
    points = (x, y, z)
    # compute angle vector pointing the origin
    c = np.linspace(-image_size // 2, image_size // 2, image_size)
    qx, qy, qz = np.meshgrid(c, c, c)
    # compute the azimuth and elevation changed coordinates
    x_render = qx * np.cos(elevation) * np.cos(azimuth) - qy * np.cos(elevation) * np.sin(azimuth) + qz * np.sin(
        elevation)
    y_render = qx * np.sin(azimuth) + qy * np.cos(azimuth)
    z_render = -qx * np.sin(elevation) * np.cos(azimuth) + qy * np.sin(elevation) * np.sin(azimuth) + qz * np.cos(
        elevation)
    # compute new grid in the x, y, z render cords
    points_render = np.array([x_render.ravel(), y_render.ravel(), z_render.ravel()]).T
    # get transfer function
    transferFunction = generate_transfer_function(values, colors, alphas)
    # compute the new volume
    image_render = np.zeros((image_size, image_size, 3))
    volume_render = interpn(points, volume, points_render, method='linear', bounds_error=False, fill_value=0).reshape(image_size, image_size, image_size)
    # TODO: assert data has no negative values
    img_aux = np.zeros((image_size, image_size, 3))  # auxiliary image to allocate memory
    a_aux: ndarray[Any, dtype[floating[_64Bit] | float_]] = np.zeros((image_size, image_size, 3))  # auxiliary image to allocate memory
    for data_slice in volume_render:
        r, g, b, a = transferFunction(np.log(data_slice))
        img_aux[:, :, 0] = r
        img_aux[:, :, 1] = g
        img_aux[:, :, 2] = b
        a_aux[:, :, 0] = a
        a_aux[:, :, 1] = a
        a_aux[:, :, 2] = a
        image_render = (1 - a_aux) * image_render + a_aux * img_aux

    print('------>>>> ', image_render.max())
    # image_render = np.clip(image_render, 0, 1)
    image_render = (image_render - image_render.min()) / (image_render.max()-image_render.min())
    # plot image render
    if ax is None:
        fig, ax = plt.subplots()
    ax.imshow(image_render)
    plt.axis('off')
    return ax
