"""Helper methods for model interpretation in general."""

import numpy
import matplotlib.colors
from matplotlib import pyplot
from scipy.ndimage.filters import gaussian_filter

DEFAULT_HORIZ_CBAR_PADDING = 0.01
DEFAULT_VERTICAL_CBAR_PADDING = 0.04


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


def plot_colour_bar(
        axes_object_or_matrix, data_values, colour_map_object,
        colour_norm_object, plot_horizontal, padding=None,
        plot_min_arrow=True, plot_max_arrow=True, fraction_of_axis_length=1.,
        font_size=30):
    """Plots colour bar on existing axes.

    :param axes_object_or_matrix: Axes handle or numpy array thereof.  Each
        handle should be an instance of `matplotlib.axes._subplots.AxesSubplot`.
    :param data_values: numpy array of data values to which the colour bar
        applies.
    :param colour_map_object: Colour scheme (instance of `matplotlib.pyplot.cm`
        or similar).
    :param colour_norm_object: Normalizer (maps from data space to colour-bar
        space, which ranges from 0...1).  Should be an instance of
        `matplotlib.colors.Normalize` or similar.
    :param plot_horizontal: Boolean flag.  If True, colour bar will be
        horizontal and below axes.  If False, will be vertical and to the right
        of axes.
    :param padding: Space between colour bar and axes (in range 0...1).
    :param plot_min_arrow: Boolean flag.  If True, will plot arrow at lower end
        of colour bar, to indicate that lower values are possible.
    :param plot_max_arrow: Same but for upper end of colour bar.
    :param fraction_of_axis_length: Determines length of colour bar.  For
        example, if `plot_horizontal == True` and
        `fraction_of_axis_length = 0.9`, will take up 90% of possible horizontal
        space.
    :param font_size: Font size.
    :return: colour_bar_object: Colour-bar handle (instance of
        `matplotlib.pyplot.colorbar`).
    """

    # Process input args.
    plot_horizontal = bool(plot_horizontal)
    plot_min_arrow = bool(plot_min_arrow)
    plot_max_arrow = bool(plot_max_arrow)
    assert fraction_of_axis_length > 0.

    if plot_min_arrow and plot_max_arrow:
        extend_arg = 'both'
    elif plot_min_arrow:
        extend_arg = 'min'
    elif plot_max_arrow:
        extend_arg = 'max'
    else:
        extend_arg = 'neither'

    if padding is None:
        if plot_horizontal:
            padding = DEFAULT_HORIZ_CBAR_PADDING
        else:
            padding = DEFAULT_VERTICAL_CBAR_PADDING

    assert padding >= 0.

    scalar_mappable_object = pyplot.cm.ScalarMappable(
        cmap=colour_map_object, norm=colour_norm_object
    )
    scalar_mappable_object.set_array(data_values)

    if isinstance(axes_object_or_matrix, numpy.ndarray):
        axes_arg = axes_object_or_matrix.ravel().tolist()
    else:
        axes_arg = axes_object_or_matrix

    colour_bar_object = pyplot.colorbar(
        ax=axes_arg, mappable=scalar_mappable_object,
        orientation='horizontal' if plot_horizontal else 'vertical',
        pad=padding, extend=extend_arg, shrink=fraction_of_axis_length
    )

    colour_bar_object.ax.tick_params(labelsize=font_size)

    if plot_horizontal:
        colour_bar_object.ax.set_xticklabels(
            colour_bar_object.ax.get_xticklabels(), rotation=90
        )

    return colour_bar_object


def plot_linear_colour_bar(
        axes_object_or_matrix, data_values, colour_map_object,
        min_value, max_value, plot_horizontal, padding=None,
        plot_min_arrow=True, plot_max_arrow=True, fraction_of_axis_length=1.,
        font_size=30):
    """Plots colour bar with linear scale.

    :param axes_object_or_matrix: See doc for `plot_colour_bar`.
    :param data_values: Same.
    :param colour_map_object: Same.
    :param min_value: Minimum value in colour bar.
    :param max_value: Max value in colour bar.
    :param plot_horizontal: See doc for `plot_colour_bar`.
    :param padding: Same.
    :param plot_min_arrow: Same.
    :param plot_max_arrow: Same.
    :param fraction_of_axis_length: Same.
    :param font_size: Same.
    """

    assert max_value > min_value
    colour_norm_object = matplotlib.colors.Normalize(
        vmin=min_value, vmax=max_value, clip=False
    )

    return plot_colour_bar(
        axes_object_or_matrix=axes_object_or_matrix, data_values=data_values,
        colour_map_object=colour_map_object,
        colour_norm_object=colour_norm_object,
        plot_horizontal=plot_horizontal, padding=padding,
        plot_min_arrow=plot_min_arrow, plot_max_arrow=plot_max_arrow,
        fraction_of_axis_length=fraction_of_axis_length, font_size=font_size
    )
