# functions to process volumes
import numpy as np

from scipy.interpolate import RegularGridInterpolator
import matplotlib.pyplot as plt
def interpolate_volume(volume: np.ndarray, factor: tuple[int, int, int] | int, spacing: tuple[float, float, float] | float=1., binary: bool=False, ax: plt.Axes = None) -> np.ndarray:
    """
    Interpolate a volume to a new shape.

    :param factor: factor int or tuple of factors
    :param volume: volume to interpolate
    :return: interpolated volume
    :rtype: np.ndarray
    """
    # convert factor to tuple if int
    if isinstance(factor, int):
        factor = (factor, factor, factor)
    factor = np.array(factor)
    if isinstance(spacing, float):
        spacing = (spacing, spacing, spacing)
    spacing = np.array(spacing)
    zoom_factor = np.r_[factor]
    # compute x,y,z coordinates of input volume
    x, y, z = np.r_[:volume.shape[0]], np.r_[:volume.shape[1]], np.r_[:volume.shape[2]]
    x = x * spacing[0]
    y = y * spacing[1]
    z = z * spacing[2]
    # create interpolation function
    interpolating_function = RegularGridInterpolator((x, y, z), volume, bounds_error=False, fill_value=0.0)
    # create new coordinates
    Nx_interp, Ny_interp, Nz_interp = np.array(volume.shape) * zoom_factor
    x_new, y_new, z_new = np.mgrid[:Nx_interp, :Ny_interp, :Nz_interp]
    spacing_new = spacing / zoom_factor
    x_new = x_new * spacing_new[0]
    y_new = y_new * spacing_new[1]
    z_new = z_new * spacing_new[2]
    new_cords = np.c_[x_new.ravel(), y_new.ravel(), z_new.ravel()]
    # create NEW data
    new_data = interpolating_function(new_cords)
    new_data = new_data.reshape((Nx_interp, Ny_interp, Nz_interp))
    # plot new data
    if ax is not None:
        N = new_data.shape[0] // 2
        ax.imshow(new_data[N, :, :])
        # plt.show()
    if binary:
        # converts new data to binary 0. to 1. range
        new_data[new_data < 0.5] = 0.
        new_data[new_data >= 0.5] = 1.
    return new_data
