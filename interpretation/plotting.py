"""Plotting methods."""

import numpy
import matplotlib.colors
from matplotlib import pyplot
from interpretation import utils

METRES_PER_SECOND_TO_KT = 3.6 / 1.852

DEFAULT_HORIZ_CBAR_PADDING = 0.01
DEFAULT_VERTICAL_CBAR_PADDING = 0.04
DEFAULT_CBAR_FONT_SIZE = 20

DEFAULT_FIG_WIDTH_INCHES = 10
DEFAULT_FIG_HEIGHT_INCHES = 10

THIS_COLOUR_LIST = [
    numpy.array([4, 233, 231]), numpy.array([1, 159, 244]),
    numpy.array([3, 0, 244]), numpy.array([2, 253, 2]),
    numpy.array([1, 197, 1]), numpy.array([0, 142, 0]),
    numpy.array([253, 248, 2]), numpy.array([229, 188, 0]),
    numpy.array([253, 149, 0]), numpy.array([253, 0, 0]),
    numpy.array([212, 0, 0]), numpy.array([188, 0, 0]),
    numpy.array([248, 0, 253]), numpy.array([152, 84, 198])
]

for p in range(len(THIS_COLOUR_LIST)):
    THIS_COLOUR_LIST[p] = THIS_COLOUR_LIST[p].astype(float) / 255

REFL_COLOUR_MAP_OBJECT = matplotlib.colors.ListedColormap(THIS_COLOUR_LIST)
REFL_COLOUR_MAP_OBJECT.set_under(numpy.ones(3))

THESE_COLOUR_BOUNDS = numpy.array([
    0.1, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70
])
REFL_COLOUR_NORM_OBJECT = matplotlib.colors.BoundaryNorm(
    THESE_COLOUR_BOUNDS, REFL_COLOUR_MAP_OBJECT.N
)

PREDICTOR_TO_COLOUR_MAP_DICT = {
    utils.TEMPERATURE_NAME: pyplot.get_cmap('YlOrRd'),
    utils.REFLECTIVITY_NAME: REFL_COLOUR_MAP_OBJECT,
    utils.U_WIND_NAME: pyplot.get_cmap('seismic'),
    utils.V_WIND_NAME: pyplot.get_cmap('seismic')
}


def _create_paneled_figure(
        num_rows, num_columns, figure_width_inches=DEFAULT_FIG_WIDTH_INCHES,
        figure_height_inches=DEFAULT_FIG_HEIGHT_INCHES,
        horizontal_spacing=0.075, vertical_spacing=0.075,
        shared_x_axis=False, shared_y_axis=False, keep_aspect_ratio=True):
    """Creates paneled figure.

    J = number of panel rows
    K = number of panel columns

    :param num_rows: J in the above discussion.
    :param num_columns: K in the above discussion.
    :param figure_width_inches: Width of the entire figure (including all
        panels).
    :param figure_height_inches: Height of the entire figure (including all
        panels).
    :param horizontal_spacing: Spacing (in figure-relative coordinates, from
        0...1) between adjacent panel columns.
    :param vertical_spacing: Spacing (in figure-relative coordinates, from
        0...1) between adjacent panel rows.
    :param shared_x_axis: Boolean flag.  If True, all panels will share the same
        x-axis.
    :param shared_y_axis: Boolean flag.  If True, all panels will share the same
        y-axis.
    :param keep_aspect_ratio: Boolean flag.  If True, the aspect ratio of each
        panel will be preserved (reflect the aspect ratio of the data plotted
        therein).
    :return: figure_object: Figure handle (instance of
        `matplotlib.figure.Figure`).
    :return: axes_object_matrix: J-by-K numpy array of axes handles (instances
        of `matplotlib.axes._subplots.AxesSubplot`).
    """

    figure_object, axes_object_matrix = pyplot.subplots(
        num_rows, num_columns, sharex=shared_x_axis, sharey=shared_y_axis,
        figsize=(figure_width_inches, figure_height_inches)
    )

    if num_rows == num_columns == 1:
        axes_object_matrix = numpy.full(
            (1, 1), axes_object_matrix, dtype=object
        )

    if num_rows == 1 or num_columns == 1:
        axes_object_matrix = numpy.reshape(
            axes_object_matrix, (num_rows, num_columns)
        )

    pyplot.subplots_adjust(
        left=0.02, bottom=0.02, right=0.98, top=0.95,
        hspace=horizontal_spacing, wspace=vertical_spacing
    )

    if not keep_aspect_ratio:
        return figure_object, axes_object_matrix

    for i in range(num_rows):
        for j in range(num_columns):
            axes_object_matrix[i][j].set(aspect='equal')

    return figure_object, axes_object_matrix


def plot_colour_bar(
        axes_object_or_matrix, data_values, colour_map_object,
        colour_norm_object, plot_horizontal, padding=None,
        plot_min_arrow=True, plot_max_arrow=True, fraction_of_axis_length=1.,
        font_size=DEFAULT_CBAR_FONT_SIZE):
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
        font_size=DEFAULT_CBAR_FONT_SIZE):
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


def plot_scalar_field_2d(
        predictor_matrix, colour_map_object, colour_norm_object=None,
        min_colour_value=None, max_colour_value=None, axes_object=None):
    """Plots scalar field (not wind) on 2-D grid.

    If `colour_norm_object is None`, both `min_colour_value` and
    `max_colour_value` must be specified.

    M = number of rows in grid
    N = number of columns in grid

    :param predictor_matrix: M-by-N numpy array of predictor values.
    :param colour_map_object: Instance of `matplotlib.pyplot.cm`.
    :param min_colour_value: Minimum value in colour scheme.
    :param max_colour_value: Max value in colour scheme.
    :param axes_object: Instance of `matplotlib.axes._subplots.AxesSubplot`.
        Will plot on these axes.  If None, will create new axes.
    :return: axes_object: See input doc.
    """

    if axes_object is None:
        _, axes_object = pyplot.subplots(
            1, 1, figsize=(DEFAULT_FIG_WIDTH_INCHES, DEFAULT_FIG_HEIGHT_INCHES)
        )

    if colour_norm_object is not None:
        min_colour_value = colour_norm_object.boundaries[0]
        max_colour_value = colour_norm_object.boundaries[-1]

    axes_object.pcolormesh(
        predictor_matrix, cmap=colour_map_object, norm=colour_norm_object,
        vmin=min_colour_value, vmax=max_colour_value,
        shading='flat', edgecolors='None'
    )

    axes_object.set_xticks([])
    axes_object.set_yticks([])

    return axes_object


def plot_wind_2d(u_wind_matrix_m_s01, v_wind_matrix_m_s01, axes_object=None):
    """Plots wind velocity on 2-D grid.

    M = number of rows in grid
    N = number of columns in grid

    :param u_wind_matrix_m_s01: M-by-N numpy array of eastward components
        (metres per second).
    :param v_wind_matrix_m_s01: M-by-N numpy array of northward components
        (metres per second).
    :param axes_object: Instance of `matplotlib.axes._subplots.AxesSubplot`.
        Will plot on these axes.
    """

    # TODO(thunderhoser): Simplify meshgrid code.
    if axes_object is None:
        _, axes_object = pyplot.subplots(
            1, 1, figsize=(DEFAULT_FIG_WIDTH_INCHES, DEFAULT_FIG_HEIGHT_INCHES)
        )

    num_grid_rows = u_wind_matrix_m_s01.shape[0]
    num_grid_columns = u_wind_matrix_m_s01.shape[1]

    x_coords_unique = numpy.linspace(
        0, num_grid_columns, num=num_grid_columns + 1, dtype=float
    )
    x_coords_unique = x_coords_unique[:-1]
    x_coords_unique = x_coords_unique + numpy.diff(x_coords_unique[:2]) / 2

    y_coords_unique = numpy.linspace(
        0, num_grid_rows, num=num_grid_rows + 1, dtype=float
    )
    y_coords_unique = y_coords_unique[:-1]
    y_coords_unique = y_coords_unique + numpy.diff(y_coords_unique[:2]) / 2

    x_coord_matrix, y_coord_matrix = numpy.meshgrid(
        x_coords_unique, y_coords_unique
    )

    speed_matrix_m_s01 = numpy.sqrt(
        u_wind_matrix_m_s01 ** 2 + v_wind_matrix_m_s01 ** 2
    )

    axes_object.barbs(
        x_coord_matrix, y_coord_matrix,
        u_wind_matrix_m_s01 * METRES_PER_SECOND_TO_KT,
        v_wind_matrix_m_s01 * METRES_PER_SECOND_TO_KT,
        speed_matrix_m_s01 * METRES_PER_SECOND_TO_KT,
        color='k', length=6, sizes={'emptybarb': 0.1},
        fill_empty=True, rounding=False
    )

    axes_object.set_xlim(0, num_grid_columns)
    axes_object.set_ylim(0, num_grid_rows)


def plot_many_predictors_with_barbs(
        predictor_matrix, predictor_names, min_colour_temp_kelvins,
        max_colour_temp_kelvins):
    """Plots many predictor variables on 2-D grid with wind barbs overlain.

    M = number of rows in grid
    N = number of columns in grid
    C = number of predictors

    :param predictor_matrix: M-by-N-by-C numpy array of predictor values.
    :param predictor_names: length-C list of predictor names.
    :param min_colour_temp_kelvins: Minimum value in temperature colour scheme.
    :param max_colour_temp_kelvins: Max value in temperature colour scheme.
    :return: figure_object: See doc for `_init_figure_panels`.
    :return: axes_object_matrix: Same.
    """

    u_wind_matrix_m_s01 = predictor_matrix[
        ..., predictor_names.index(utils.U_WIND_NAME)
    ]
    v_wind_matrix_m_s01 = predictor_matrix[
        ..., predictor_names.index(utils.V_WIND_NAME)
    ]

    non_wind_predictor_names = [
        p for p in predictor_names
        if p not in [utils.U_WIND_NAME, utils.V_WIND_NAME]
    ]

    figure_object, axes_object_matrix = _create_paneled_figure(
        num_rows=1, num_columns=len(non_wind_predictor_names),
    )

    for k in range(len(non_wind_predictor_names)):
        this_predictor_index = predictor_names.index(
            non_wind_predictor_names[k]
        )

        if non_wind_predictor_names[k] == utils.REFLECTIVITY_NAME:
            this_colour_norm_object = REFL_COLOUR_NORM_OBJECT
            this_min_colour_value = None
            this_max_colour_value = None
        else:
            this_colour_norm_object = None
            this_min_colour_value = min_colour_temp_kelvins + 0.
            this_max_colour_value = max_colour_temp_kelvins + 0.

        this_colour_map_object = PREDICTOR_TO_COLOUR_MAP_DICT[
            non_wind_predictor_names[k]
        ]

        plot_scalar_field_2d(
            predictor_matrix=predictor_matrix[..., this_predictor_index],
            colour_map_object=this_colour_map_object,
            colour_norm_object=this_colour_norm_object,
            min_colour_value=this_min_colour_value,
            max_colour_value=this_max_colour_value,
            axes_object=axes_object_matrix[0, k]
        )

        if non_wind_predictor_names[k] == utils.REFLECTIVITY_NAME:
            this_colour_bar_object = plot_colour_bar(
                axes_object_or_matrix=axes_object_matrix[0, k],
                data_values=predictor_matrix[..., this_predictor_index],
                colour_map_object=this_colour_map_object,
                colour_norm_object=this_colour_norm_object,
                plot_horizontal=True, plot_min_arrow=False, plot_max_arrow=True
            )
        else:
            this_colour_bar_object = plot_linear_colour_bar(
                axes_object_or_matrix=axes_object_matrix[0, k],
                data_values=predictor_matrix[..., this_predictor_index],
                colour_map_object=this_colour_map_object,
                min_value=this_min_colour_value,
                max_value=this_max_colour_value,
                plot_horizontal=True, plot_min_arrow=True, plot_max_arrow=True
            )

        plot_wind_2d(
            u_wind_matrix_m_s01=u_wind_matrix_m_s01,
            v_wind_matrix_m_s01=v_wind_matrix_m_s01,
            axes_object=axes_object_matrix[0, k]
        )

        this_colour_bar_object.set_label(
            non_wind_predictor_names[k], fontsize=DEFAULT_CBAR_FONT_SIZE
        )

    return figure_object, axes_object_matrix


def plot_many_predictors_sans_barbs(
        predictor_matrix, predictor_names, min_colour_temp_kelvins,
        max_colour_temp_kelvins, max_colour_wind_speed_m_s01):
    """Plots many predictor variables on 2-D grid; no wind barbs overlain.

    In this case, both u-wind and v-wind are plotted as separate maps.

    M = number of rows in grid
    N = number of columns in grid
    C = number of predictors

    :param predictor_matrix: M-by-N-by-C numpy array of predictor values.
    :param predictor_names: length-C list of predictor names.
    :param min_colour_temp_kelvins: Minimum value in temperature colour scheme.
    :param max_colour_temp_kelvins: Max value in temperature colour scheme.
    :param max_colour_wind_speed_m_s01: Max wind speed (metres per second) in
        colour maps for both u- and v-components.  The minimum wind speed be
        `-1 * max_colour_wind_speed_m_s01`, so the diverging colour scheme will
        be zero-centered.
    :return: figure_object: See doc for `_init_figure_panels`.
    :return: axes_object_matrix: Same.
    """

    num_predictors = len(predictor_names)
    num_panel_rows = int(numpy.floor(
        numpy.sqrt(num_predictors)
    ))
    num_panel_columns = int(numpy.ceil(
        float(num_predictors) / num_panel_rows
    ))

    figure_object, axes_object_matrix = _create_paneled_figure(
        num_rows=num_panel_rows, num_columns=num_panel_columns,
        horizontal_spacing=0.15, vertical_spacing=0.15
    )

    for i in range(num_panel_rows):
        for j in range(num_panel_columns):
            k = i * num_panel_columns + j
            if k >= num_predictors:
                break

            this_colour_map_object = PREDICTOR_TO_COLOUR_MAP_DICT[
                predictor_names[k]
            ]

            if predictor_names[k] == utils.REFLECTIVITY_NAME:
                this_colour_norm_object = REFL_COLOUR_NORM_OBJECT
                this_min_colour_value = None
                this_max_colour_value = None
            elif predictor_names[k] == utils.TEMPERATURE_NAME:
                this_colour_norm_object = None
                this_min_colour_value = min_colour_temp_kelvins + 0.
                this_max_colour_value = max_colour_temp_kelvins + 0.
            else:
                this_colour_norm_object = None
                this_min_colour_value = -1 * max_colour_wind_speed_m_s01
                this_max_colour_value = max_colour_wind_speed_m_s01 + 0.

            plot_scalar_field_2d(
                predictor_matrix=predictor_matrix[..., k],
                colour_map_object=this_colour_map_object,
                colour_norm_object=this_colour_norm_object,
                min_colour_value=this_min_colour_value,
                max_colour_value=this_max_colour_value,
                axes_object=axes_object_matrix[i, j]
            )

            if predictor_names[k] == utils.REFLECTIVITY_NAME:
                this_colour_bar_object = plot_colour_bar(
                    axes_object_or_matrix=axes_object_matrix[i, j],
                    data_values=predictor_matrix[..., k],
                    colour_map_object=this_colour_map_object,
                    colour_norm_object=this_colour_norm_object,
                    plot_horizontal=True, fraction_of_axis_length=0.85,
                    plot_min_arrow=False, plot_max_arrow=True
                )
            else:
                this_colour_bar_object = plot_linear_colour_bar(
                    axes_object_or_matrix=axes_object_matrix[i, j],
                    data_values=predictor_matrix[..., k],
                    colour_map_object=this_colour_map_object,
                    min_value=this_min_colour_value,
                    max_value=this_max_colour_value,
                    plot_horizontal=True, fraction_of_axis_length=0.85,
                    plot_min_arrow=True, plot_max_arrow=True
                )

            this_colour_bar_object.set_label(
                predictor_names[k], fontsize=DEFAULT_CBAR_FONT_SIZE
            )

    return figure_object, axes_object_matrix
