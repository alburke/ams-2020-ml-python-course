"""Helper methods for model interpretation in general."""

from scipy.ndimage.filters import gaussian_filter


def apply_gaussian_filter(input_matrix, e_folding_radius_grid_cells):
    """Applies Gaussian filter to any-dimensional grid.

    :param input_matrix: numpy array with any dimensions.
    :param e_folding_radius_grid_cells: e-folding radius (num grid cells).
    :return: output_matrix: numpy array after smoothing (same dimensions as
        input).
    """

    assert e_folding_radius_grid_cells >= 0.
    return gaussian_filter(
        input_matrix, sigma=e_folding_radius_grid_cells, order=0, mode='nearest'
    )
