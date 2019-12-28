"""Helper methods for class-activation maps."""

import numpy
from keras import backend as K
import tensorflow
from scipy.interpolate import (
    UnivariateSpline, RectBivariateSpline, RegularGridInterpolator
)
from interpretation import utils
from interpretation.saliency import _get_grid_points

DEFAULT_LINE_WIDTH = 2.


def _compute_gradients(loss_tensor, list_of_input_tensors):
    """Computes gradient of each input tensor with respect to loss tensor.

    T = number of tensors

    :param loss_tensor: Loss tensor.
    :param list_of_input_tensors: length-T list of input tensors.
    :return: list_of_gradient_tensors: length-T list of gradient tensors.
    """

    list_of_gradient_tensors = tensorflow.gradients(
        loss_tensor, list_of_input_tensors
    )

    for i in range(len(list_of_gradient_tensors)):
        if list_of_gradient_tensors[i] is not None:
            continue

        list_of_gradient_tensors[i] = tensorflow.zeros_like(
            list_of_input_tensors[i]
        )

    return list_of_gradient_tensors


def _normalize_tensor(input_tensor):
    """Normalizes tensor to Euclidean magnitude (or "L_2 norm") of 1.0.

    :param input_tensor: Input tensor.
    :return: output_tensor: Same as input but with Euclidean magnitude of 1.0.
    """

    rms_tensor = K.sqrt(K.mean(K.square(input_tensor)))
    return input_tensor / (rms_tensor + K.epsilon())


def _upsample_cam(class_activation_matrix, new_dimensions):
    """Upsamples class-activation map (CAM).

    The CAM may be 1-, 2-, or 3-dimensional.

    :param class_activation_matrix: numpy array of class activations.
    :param new_dimensions: numpy array of new dimensions.  If
        `class_activation_matrix` is N-dimensional, this array must be length-N.
    :return: class_activation_matrix: Upsampled version of input.
    """

    num_rows_new = new_dimensions[0]
    row_indices_new = numpy.linspace(
        1, num_rows_new, num=num_rows_new, dtype=float
    )
    row_indices_orig = numpy.linspace(
        1, num_rows_new, num=class_activation_matrix.shape[0], dtype=float
    )

    if len(new_dimensions) == 1:
        interp_object = UnivariateSpline(
            x=row_indices_orig, y=numpy.ravel(class_activation_matrix),
            k=3, s=0
        )

        return interp_object(row_indices_new)

    num_columns_new = new_dimensions[1]
    column_indices_new = numpy.linspace(
        1, num_columns_new, num=num_columns_new, dtype=float
    )
    column_indices_orig = numpy.linspace(
        1, num_columns_new, num=class_activation_matrix.shape[1], dtype=float
    )

    if len(new_dimensions) == 2:
        interp_object = RectBivariateSpline(
            x=row_indices_orig, y=column_indices_orig,
            z=class_activation_matrix, kx=3, ky=3, s=0
        )

        return interp_object(x=row_indices_new, y=column_indices_new, grid=True)

    num_heights_new = new_dimensions[2]
    height_indices_new = numpy.linspace(
        1, num_heights_new, num=num_heights_new, dtype=float
    )
    height_indices_orig = numpy.linspace(
        1, num_heights_new, num=class_activation_matrix.shape[2], dtype=float
    )

    interp_object = RegularGridInterpolator(
        points=(row_indices_orig, column_indices_orig, height_indices_orig),
        values=class_activation_matrix, method='linear'
    )

    column_index_matrix, row_index_matrix, height_index_matrix = (
        numpy.meshgrid(column_indices_new, row_indices_new, height_indices_new)
    )
    query_point_matrix = numpy.stack(
        (row_index_matrix, column_index_matrix, height_index_matrix), axis=-1
    )

    return interp_object(query_point_matrix)


def _plot_cam_one_channel(
        class_activation_matrix_2d, axes_object, colour_map_object,
        min_contour_value, max_contour_value, contour_interval,
        line_width=DEFAULT_LINE_WIDTH):
    """Plots 2-D class-activation map with line contours.

    M = number of rows in grid
    N = number of columns in grid

    :param class_activation_matrix_2d: M-by-N numpy array of class activations.
    :param axes_object: Will plot on these axes (instance of
        `matplotlib.axes._subplots.AxesSubplot`).
    :param colour_map_object: Colour scheme (instance of `matplotlib.pyplot.cm`
        or similar).
    :param min_contour_value: Minimum contour value.
    :param max_contour_value: Max contour value.
    :param contour_interval: Interval between successive contours.
    :param line_width: Line width for contours.
    """

    # Check input args.
    assert not numpy.any(numpy.isnan(class_activation_matrix_2d))
    assert len(class_activation_matrix_2d.shape) == 2

    max_contour_value = max([
        min_contour_value + 1e-3, max_contour_value
    ])

    contour_interval = max([contour_interval, 1e-4])
    contour_interval = min([
        contour_interval, max_contour_value - min_contour_value
    ])

    num_contours = 1 + int(numpy.round(
        (max_contour_value - min_contour_value) / contour_interval
    ))
    contour_values = numpy.linspace(
        min_contour_value, max_contour_value, num=num_contours, dtype=float
    )

    # Find grid coordinates.
    num_grid_rows = class_activation_matrix_2d.shape[0]
    num_grid_columns = class_activation_matrix_2d.shape[1]
    x_coord_spacing = num_grid_columns ** -1
    y_coord_spacing = num_grid_rows ** -1

    # TODO(thunderhoser): Calling private method here is a HACK.
    x_coords, y_coords = _get_grid_points(
        x_min=x_coord_spacing / 2, y_min=y_coord_spacing / 2,
        x_spacing=x_coord_spacing, y_spacing=y_coord_spacing,
        num_rows=num_grid_rows, num_columns=num_grid_columns
    )

    x_coord_matrix, y_coord_matrix = numpy.meshgrid(x_coords, y_coords)

    # Plot contours.
    axes_object.contour(
        x_coord_matrix, y_coord_matrix, class_activation_matrix_2d,
        contour_values, cmap=colour_map_object,
        vmin=numpy.min(contour_values), vmax=numpy.max(contour_values),
        linewidths=line_width, linestyles='solid', zorder=1e6,
        transform=axes_object.transAxes
    )


def run_gradcam(model_object, input_matrix, target_class, target_layer_name):
    """Runs Grad-CAM (gradient-weighted class-activation-mapping).

    :param model_object: Trained model (instance of `keras.models.Model` or
        `keras.models.Sequential`).
    :param input_matrix: numpy array of inputs (predictors) for one example.
    :param target_class: Target class.  Class-activation maps will be created
        for the [k + 1]th class, where k = `target_class`.
    :param target_layer_name: Name of target layer.  Neuron-importance weights
        will be based on activations in this layer.
    :return: class_activation_matrix: numpy array of class activations.  This
        array will have the same dimensions as `input_matrix` but without the
        final axis.  For example, if `input_matrix` is 32 x 32 x 4
        (32 rows x 32 columns x 4 channels), `class_activation_matrix` will be
        32 x 32.
    """

    # Check input args.
    target_class = int(numpy.round(target_class))
    assert target_class >= 0

    assert not numpy.any(numpy.isnan(input_matrix))
    num_spatial_dim = len(input_matrix.shape) - 1
    assert 1 <= num_spatial_dim <= 3

    # Create loss tensor.
    output_layer_object = model_object.layers[-1].output
    num_output_neurons = output_layer_object.get_shape().as_list()[-1]

    if num_output_neurons == 1:
        assert target_class <= 1

        if target_class == 1:
            loss_tensor = model_object.layers[-1].input[..., 0]
        else:
            loss_tensor = -1 * model_object.layers[-1].input[..., 0]
    else:
        assert target_class < num_output_neurons
        loss_tensor = model_object.layers[-1].input[..., target_class]

    # Create gradient function.
    target_layer_activation_tensor = model_object.get_layer(
        name=target_layer_name
    ).output

    gradient_tensor = _compute_gradients(
        loss_tensor, [target_layer_activation_tensor]
    )[0]
    gradient_tensor = _normalize_tensor(gradient_tensor)

    if isinstance(model_object.input, list):
        input_tensor = model_object.input[0]
    else:
        input_tensor = model_object.input

    gradient_function = K.function(
        [input_tensor],
        [target_layer_activation_tensor, gradient_tensor]
    )

    # Evaluate gradient function.
    input_matrix_with_example_axis = numpy.expand_dims(input_matrix, axis=0)
    target_layer_activation_matrix, gradient_matrix = gradient_function(
        [input_matrix_with_example_axis]
    )

    target_layer_activation_matrix = target_layer_activation_matrix[0, ...]
    gradient_matrix = gradient_matrix[0, ...]

    # Compute class-activation map.
    these_axes = [i for i in range(num_spatial_dim)]
    mean_weight_by_filter = numpy.mean(gradient_matrix, axis=tuple(these_axes))

    class_activation_matrix = numpy.ones(
        target_layer_activation_matrix.shape[:-1]
    )
    num_filters = len(mean_weight_by_filter)

    for k in range(num_filters):
        class_activation_matrix += (
            mean_weight_by_filter[k] * target_layer_activation_matrix[..., k]
        )

    # Upsample class-activation map to input space.
    input_spatial_dim = numpy.array(input_matrix.shape[:-1], dtype=int)
    class_activation_matrix = _upsample_cam(
        class_activation_matrix=class_activation_matrix,
        new_dimensions=input_spatial_dim
    )

    return numpy.maximum(class_activation_matrix, 0.)


def smooth_cams(class_activation_matrix, smoothing_radius_grid_cells):
    """Smooths class-activation maps for many examples.

    E = number of examples
    D = number of spatial dimensions

    :param class_activation_matrix: numpy array with class-activation maps for
        one or more examples.  Should have D + 1 dimensions, and the first axis
        should have length E.
    :param smoothing_radius_grid_cells: e-folding radius (number of grid cells).
    :return: saliency_matrices: Smoothed version of input.
    """

    num_examples = class_activation_matrix.shape[0]

    for i in range(num_examples):
        class_activation_matrix[i, ...] = utils.apply_gaussian_filter(
            input_matrix=class_activation_matrix[i, ...],
            e_folding_radius_grid_cells=smoothing_radius_grid_cells
        )

    return class_activation_matrix


def plot_2d_cam(
        class_activation_matrix_2d, axes_object_matrix, num_channels,
        colour_map_object, min_contour_value, max_contour_value,
        contour_interval, line_width=DEFAULT_LINE_WIDTH):
    """Plots 2-D class-activation map for one example.

    :param class_activation_matrix_2d: See doc for `_plot_cam_one_channel`.
    :param axes_object_matrix: 2-D numpy array of axes (each an instance of
        `matplotlib.axes._subplots.AxesSubplot`).
    :param num_channels: Number of channels (the same CAM will be plotted on top
        of each channel).
    :param colour_map_object: See doc for `_plot_cam_one_channel`.
    :param min_contour_value: Same.
    :param max_contour_value: Same.
    :param contour_interval: Same.
    :param line_width: Same.
    """

    num_panel_rows = axes_object_matrix.shape[0]
    num_panel_columns = axes_object_matrix.shape[1]

    for k in range(num_channels):
        i, j = numpy.unravel_index(k, (num_panel_rows, num_panel_columns))
        this_axes_object = axes_object_matrix[i, j]

        _plot_cam_one_channel(
            class_activation_matrix_2d=class_activation_matrix_2d,
            axes_object=this_axes_object,
            colour_map_object=colour_map_object,
            min_contour_value=min_contour_value,
            max_contour_value=max_contour_value,
            contour_interval=contour_interval, line_width=line_width
        )
