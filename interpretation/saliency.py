"""Helper methods for saliency."""

import numpy
from keras import backend as K
from interpretation import utils

DEFAULT_LINE_WIDTH = 2.


def _do_saliency_calculations(
        model_object, loss_tensor, list_of_input_matrices):
    """Does saliency calculations.

    T = number of input tensors to the model
    E = number of examples (storm objects)

    :param model_object: Instance of `keras.models.Model`.
    :param loss_tensor: Keras tensor defining the loss function.
    :param list_of_input_matrices: length-T list of numpy arrays, comprising one
        or more examples (storm objects).  list_of_input_matrices[i] must have
        the same dimensions as the [i]th input tensor to the model.
    :return: list_of_saliency_matrices: length-T list of numpy arrays,
        comprising the saliency map for each example.
        list_of_saliency_matrices[i] has the same dimensions as
        list_of_input_matrices[i] and defines the "saliency" of each value x,
        which is the gradient of the loss function with respect to x.
    """

    if isinstance(model_object.input, list):
        list_of_input_tensors = model_object.input
    else:
        list_of_input_tensors = [model_object.input]

    list_of_gradient_tensors = K.gradients(loss_tensor, list_of_input_tensors)
    num_input_tensors = len(list_of_input_tensors)

    for i in range(num_input_tensors):
        list_of_gradient_tensors[i] /= K.maximum(
            K.std(list_of_gradient_tensors[i]),
            K.epsilon()
        )

    inputs_to_gradients_function = K.function(
        list_of_input_tensors + [K.learning_phase()],
        list_of_gradient_tensors
    )

    list_of_saliency_matrices = inputs_to_gradients_function(
        list_of_input_matrices + [0]
    )

    for i in range(num_input_tensors):
        list_of_saliency_matrices[i] *= -1

    return list_of_saliency_matrices


def _get_grid_points(x_min, x_spacing, num_columns, y_min, y_spacing, num_rows):
    """Returns grid points in regular x-y grid.

    M = number of rows in grid
    N = number of columns in grid

    :param x_min: Minimum x-coordinate over all grid points.
    :param x_spacing: Spacing between adjacent grid points in x-direction.
    :param num_columns: N in the above definition.
    :param y_min: Minimum y-coordinate over all grid points.
    :param y_spacing: Spacing between adjacent grid points in y-direction.
    :param num_rows: M in the above definition.
    :return: x_coords: length-N numpy array with x-coordinates at grid points.
    :return: y_coords: length-M numpy array with y-coordinates at grid points.
    """

    # TODO(thunderhoser): Put this in utils.py.

    x_max = x_min + (num_columns - 1) * x_spacing
    y_max = y_min + (num_rows - 1) * y_spacing

    x_coords = numpy.linspace(x_min, x_max, num=num_columns)
    y_coords = numpy.linspace(y_min, y_max, num=num_rows)

    return x_coords, y_coords


def _plot_2d_saliency_map(
        saliency_matrix_2d, axes_object, colour_map_object, max_contour_value,
        contour_interval, line_width=DEFAULT_LINE_WIDTH):
    """Plots 2-D saliency map with line contours.

    M = number of rows in grid
    N = number of columns in grid

    :param saliency_matrix_2d: M-by-N numpy array of saliency values.
    :param axes_object: Will plot on these axes (instance of
        `matplotlib.axes._subplots.AxesSubplot`).
    :param colour_map_object: Colour scheme (instance of `matplotlib.pyplot.cm`
        or similar).
    :param max_contour_value: Max contour value.  Contour values will range from
        -v...v, where v = `max_contour_value`.
    :param contour_interval: Interval between successive contours.
    :param line_width: Line width for contours.
    """

    # Check input args.
    assert max_contour_value >= 0.
    max_contour_value = max([max_contour_value, 1e-3])

    assert contour_interval >= 0.
    contour_interval = max([contour_interval, 1e-4])

    assert not numpy.any(numpy.isnan(saliency_matrix_2d))
    assert len(saliency_matrix_2d.shape) == 2
    assert contour_interval < max_contour_value

    half_num_contours = int(numpy.round(
        1 + max_contour_value / contour_interval
    ))

    # Find grid coordinates.
    num_grid_rows = saliency_matrix_2d.shape[0]
    num_grid_columns = saliency_matrix_2d.shape[1]
    x_coord_spacing = num_grid_columns ** -1
    y_coord_spacing = num_grid_rows ** -1

    x_coords, y_coords = _get_grid_points(
        x_min=x_coord_spacing / 2, y_min=y_coord_spacing / 2,
        x_spacing=x_coord_spacing, y_spacing=y_coord_spacing,
        num_rows=num_grid_rows, num_columns=num_grid_columns
    )

    x_coord_matrix, y_coord_matrix = numpy.meshgrid(x_coords, y_coords)

    # Plot positive contours.
    positive_contour_values = numpy.linspace(
        0., max_contour_value, num=half_num_contours
    )

    axes_object.contour(
        x_coord_matrix, y_coord_matrix, saliency_matrix_2d,
        positive_contour_values, cmap=colour_map_object,
        vmin=numpy.min(positive_contour_values),
        vmax=numpy.max(positive_contour_values),
        linewidths=line_width, linestyles='solid', zorder=1e6,
        transform=axes_object.transAxes
    )

    # Plot negative contours.
    negative_contour_values = positive_contour_values[1:]

    axes_object.contour(
        x_coord_matrix, y_coord_matrix, -saliency_matrix_2d,
        negative_contour_values, cmap=colour_map_object,
        vmin=numpy.min(negative_contour_values),
        vmax=numpy.max(negative_contour_values),
        linewidths=line_width, linestyles='dashed', zorder=1e6,
        transform=axes_object.transAxes
    )


def get_saliency_maps_for_class(
        model_object, target_class, list_of_input_matrices):
    """For each input example, creates saliency map for prob of target class.

    :param model_object: Trained model (instance of `keras.models.Model` or
        `keras.models.Sequential`).
    :param target_class: Saliency maps will be created for this class.  Must be
        an integer in 0...(K - 1), where K = number of classes.
    :param list_of_input_matrices: See doc for `_do_saliency_calculations`.
    :return: list_of_saliency_matrices: See doc for `_do_saliency_calculations`.
    """

    target_class = int(numpy.round(target_class))
    assert target_class >= 0

    num_output_neurons = (
        model_object.layers[-1].output.get_shape().as_list()[-1]
    )

    if num_output_neurons == 1:
        assert target_class <= 1

        if target_class == 1:
            loss_tensor = K.mean(
                (model_object.layers[-1].output[..., 0] - 1) ** 2
            )
        else:
            loss_tensor = K.mean(model_object.layers[-1].output[..., 0] ** 2)
    else:
        assert target_class < num_output_neurons

        loss_tensor = K.mean(
            (model_object.layers[-1].output[..., target_class] - 1) ** 2
        )

    return _do_saliency_calculations(
        model_object=model_object, loss_tensor=loss_tensor,
        list_of_input_matrices=list_of_input_matrices)


def smooth_saliency_maps(saliency_matrices, smoothing_radius_grid_cells):
    """Smooths saliency maps via Gaussian filter.

    T = number of input tensors to the model

    :param saliency_matrices: length-T list of numpy arrays.
    :param smoothing_radius_grid_cells: e-folding radius (number of grid cells).
    :return: saliency_matrices: Smoothed version of input.
    """

    num_matrices = len(saliency_matrices)
    num_examples = saliency_matrices[0].shape[0]

    for j in range(num_matrices):
        this_num_channels = saliency_matrices[j].shape[-1]

        for i in range(num_examples):
            for k in range(this_num_channels):
                saliency_matrices[j][i, ..., k] = utils.apply_gaussian_filter(
                    input_matrix=saliency_matrices[j][i, ..., k],
                    e_folding_radius_grid_cells=smoothing_radius_grid_cells
                )

    return saliency_matrices


def plot_saliency_maps(
        saliency_matrix_3d, axes_object_matrix, colour_map_object,
        max_contour_value, contour_interval,
        line_width=DEFAULT_LINE_WIDTH):
    """Plots many saliency maps (one for each channel).

    M = number of rows in grid
    N = number of columns in grid
    C = number of channels

    :param saliency_matrix_3d: M-by-N-by-C numpy array of saliency values.
    :param axes_object_matrix: 2-D numpy array of axes (each an instance of
        `matplotlib.axes._subplots.AxesSubplot`).
    :param colour_map_object: See doc for `_plot_2d_saliency_map`.
    :param max_contour_value: Same.
    :param contour_interval: Same.
    :param line_width: Same.
    """

    assert len(saliency_matrix_3d.shape) == 3

    num_channels = saliency_matrix_3d.shape[-1]
    num_panel_rows = axes_object_matrix.shape[0]
    num_panel_columns = axes_object_matrix.shape[1]

    for k in range(num_channels):
        i, j = numpy.unravel_index(k, (num_panel_rows, num_panel_columns))
        this_axes_object = axes_object_matrix[i, j]

        _plot_2d_saliency_map(
            saliency_matrix_2d=saliency_matrix_3d[..., k],
            axes_object=this_axes_object,
            colour_map_object=colour_map_object,
            max_contour_value=max_contour_value,
            contour_interval=contour_interval, line_width=line_width
        )
