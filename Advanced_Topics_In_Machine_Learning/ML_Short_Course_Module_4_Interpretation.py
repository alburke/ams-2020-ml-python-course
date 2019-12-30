"""Code for AMS 2019 short course."""

import copy
import glob
import errno
import random
import os.path
import json
import pickle
import time
import calendar
import numpy
import netCDF4
from scipy.interpolate import (
    UnivariateSpline, RectBivariateSpline, RegularGridInterpolator)
import keras
from keras import backend as K
import tensorflow
from tensorflow.python.framework import ops as tensorflow_ops
from sklearn.metrics import auc as scikit_learn_auc
import matplotlib.colors
import matplotlib.pyplot as pyplot
from module_4 import keras_metrics
from module_4 import roc_curves
from module_4 import performance_diagrams
from module_4 import attributes_diagrams

# Directories.
# MODULE4_DIR_NAME = '.'
# SHORT_COURSE_DIR_NAME = '..'

MODULE4_DIR_NAME = os.path.dirname(__file__)
SHORT_COURSE_DIR_NAME = os.path.dirname(MODULE4_DIR_NAME)
DEFAULT_IMAGE_DIR_NAME = '{0:s}/data/track_data_ncar_ams_3km_nc_small'.format(
    SHORT_COURSE_DIR_NAME)

# Plotting constants.
FIGURE_WIDTH_INCHES = 15
FIGURE_HEIGHT_INCHES = 15
FIGURE_RESOLUTION_DPI = 300

BAR_GRAPH_FACE_COLOUR = numpy.array([166, 206, 227], dtype=float) / 255
BAR_GRAPH_EDGE_COLOUR = numpy.full(3, 0.)
BAR_GRAPH_EDGE_WIDTH = 2.

SALIENCY_COLOUR_MAP_OBJECT = pyplot.cm.Greys

FONT_SIZE = 30
pyplot.rc('font', size=FONT_SIZE)
pyplot.rc('axes', titlesize=FONT_SIZE)
pyplot.rc('axes', labelsize=FONT_SIZE)
pyplot.rc('xtick', labelsize=FONT_SIZE)
pyplot.rc('ytick', labelsize=FONT_SIZE)
pyplot.rc('legend', fontsize=FONT_SIZE)
pyplot.rc('figure', titlesize=FONT_SIZE)

# Naming constants.
CSV_METADATA_COLUMNS = [
    'Step_ID', 'Track_ID', 'Ensemble_Name', 'Ensemble_Member', 'Run_Date',
    'Valid_Date', 'Forecast_Hour', 'Valid_Hour_UTC'
]

CSV_EXTRANEOUS_COLUMNS = [
    'Duration', 'Centroid_Lon', 'Centroid_Lat', 'Centroid_X', 'Centroid_Y',
    'Storm_Motion_U', 'Storm_Motion_V', 'Matched', 'Max_Hail_Size',
    'Num_Matches', 'Shape', 'Location', 'Scale'
]

CSV_TARGET_NAME = 'RVORT1_MAX-future_max'
TARGET_NAME = 'max_future_vorticity_s01'

NETCDF_REFL_NAME = 'REFL_COM_curr'
NETCDF_TEMP_NAME = 'T2_curr'
NETCDF_U_WIND_NAME = 'U10_curr'
NETCDF_V_WIND_NAME = 'V10_curr'
NETCDF_PREDICTOR_NAMES = [
    NETCDF_REFL_NAME, NETCDF_TEMP_NAME, NETCDF_U_WIND_NAME, NETCDF_V_WIND_NAME
]

REFLECTIVITY_NAME = 'reflectivity_dbz'
TEMPERATURE_NAME = 'temperature_kelvins'
U_WIND_NAME = 'u_wind_m_s01'
V_WIND_NAME = 'v_wind_m_s01'
PREDICTOR_NAMES = [
    REFLECTIVITY_NAME, TEMPERATURE_NAME, U_WIND_NAME, V_WIND_NAME
]

NETCDF_TRACK_ID_NAME = 'track_id'
NETCDF_TRACK_STEP_NAME = 'track_step'
NETCDF_TARGET_NAME = 'RVORT1_MAX_future'

NUM_VALUES_KEY = 'num_values'
MEAN_VALUE_KEY = 'mean_value'
MEAN_OF_SQUARES_KEY = 'mean_of_squares'

STORM_IDS_KEY = 'storm_ids'
STORM_STEPS_KEY = 'storm_steps'
PREDICTOR_NAMES_KEY = 'predictor_names'
PREDICTOR_MATRIX_KEY = 'predictor_matrix'
TARGET_NAME_KEY = 'target_name'
TARGET_MATRIX_KEY = 'target_matrix'

TRAINING_FILES_KEY = 'training_file_names'
NORMALIZATION_DICT_KEY = 'normalization_dict'
BINARIZATION_THRESHOLD_KEY = 'binarization_threshold'
NUM_EXAMPLES_PER_BATCH_KEY = 'num_examples_per_batch'
NUM_TRAINING_BATCHES_KEY = 'num_training_batches_per_epoch'
VALIDATION_FILES_KEY = 'validation_file_names'
NUM_VALIDATION_BATCHES_KEY = 'num_validation_batches_per_epoch'
CNN_FILE_KEY = 'cnn_file_name'
CNN_FEATURE_LAYER_KEY = 'cnn_feature_layer_name'

PERMUTED_PREDICTORS_KEY = 'permuted_predictor_name_by_step'
HIGHEST_COSTS_KEY = 'highest_cost_by_step'
ORIGINAL_COST_KEY = 'original_cost'
STEP1_PREDICTORS_KEY = 'predictor_names_step1'
STEP1_COSTS_KEY = 'costs_step1'

EOF_MATRIX_KEY = 'eof_matrix'
FEATURE_MEANS_KEY = 'feature_means'
FEATURE_STDEVS_KEY = 'feature_standard_deviations'

NOVEL_IMAGES_ACTUAL_KEY = 'novel_image_matrix_actual'
NOVEL_IMAGES_UPCONV_KEY = 'novel_image_matrix_upconv'
NOVEL_IMAGES_UPCONV_SVD_KEY = 'novel_image_matrix_upconv_svd'

# More plotting constants.
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

PREDICTOR_TO_COLOUR_MAP_DICT = {
    TEMPERATURE_NAME: pyplot.cm.YlOrRd,
    REFLECTIVITY_NAME: REFL_COLOUR_MAP_OBJECT,
    U_WIND_NAME: pyplot.cm.seismic,
    V_WIND_NAME: pyplot.cm.seismic
}

THESE_COLOUR_BOUNDS = numpy.array(
    [0.1, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70])
REFL_COLOUR_NORM_OBJECT = matplotlib.colors.BoundaryNorm(
    THESE_COLOUR_BOUNDS, REFL_COLOUR_MAP_OBJECT.N)

# Deep-learning constants.
L1_WEIGHT = 0.
L2_WEIGHT = 0.001
NUM_PREDICTORS_TO_FIRST_NUM_FILTERS = 8
NUM_CONV_LAYER_SETS = 2
NUM_CONV_LAYERS_PER_SET = 2
NUM_CONV_FILTER_ROWS = 3
NUM_CONV_FILTER_COLUMNS = 3
CONV_LAYER_DROPOUT_FRACTION = None
USE_BATCH_NORMALIZATION = True
SLOPE_FOR_RELU = 0.2
NUM_POOLING_ROWS = 2
NUM_POOLING_COLUMNS = 2
NUM_DENSE_LAYERS = 3
DENSE_LAYER_DROPOUT_FRACTION = 0.5

NUM_SMOOTHING_FILTER_ROWS = 5
NUM_SMOOTHING_FILTER_COLUMNS = 5

MIN_XENTROPY_DECREASE_FOR_EARLY_STOP = 0.005
MIN_MSE_DECREASE_FOR_EARLY_STOP = 0.005
NUM_EPOCHS_FOR_EARLY_STOPPING = 5

LIST_OF_METRIC_FUNCTIONS = [
    keras_metrics.accuracy, keras_metrics.binary_accuracy,
    keras_metrics.binary_csi, keras_metrics.binary_frequency_bias,
    keras_metrics.binary_pod, keras_metrics.binary_pofd,
    keras_metrics.binary_peirce_score, keras_metrics.binary_success_ratio,
    keras_metrics.binary_focn
]

METRIC_FUNCTION_DICT = {
    'accuracy': keras_metrics.accuracy,
    'binary_accuracy': keras_metrics.binary_accuracy,
    'binary_csi': keras_metrics.binary_csi,
    'binary_frequency_bias': keras_metrics.binary_frequency_bias,
    'binary_pod': keras_metrics.binary_pod,
    'binary_pofd': keras_metrics.binary_pofd,
    'binary_peirce_score': keras_metrics.binary_peirce_score,
    'binary_success_ratio': keras_metrics.binary_success_ratio,
    'binary_focn': keras_metrics.binary_focn
}

DEFAULT_NUM_BWO_ITERATIONS = 200
DEFAULT_BWO_LEARNING_RATE = 0.01

# Misc constants.
SEPARATOR_STRING = '\n\n' + '*' * 50 + '\n\n'
MINOR_SEPARATOR_STRING = '\n\n' + '-' * 50 + '\n\n'

DATE_FORMAT = '%Y%m%d'
DATE_FORMAT_REGEX = '[0-9][0-9][0-9][0-9][0-1][0-9][0-3][0-9]'

BACKPROP_FUNCTION_NAME = 'GuidedBackProp'

MIN_PROBABILITY = 1e-15
MAX_PROBABILITY = 1. - MIN_PROBABILITY
METRES_PER_SECOND_TO_KT = 3.6 / 1.852


def evaluate_cnn(
        cnn_model_object, image_dict, cnn_metadata_dict, output_dir_name):
    """Evaluates trained CNN (convolutional neural net).

    :param cnn_model_object: Trained instance of `keras.models.Model`.
    :param image_dict: Dictionary created by `read_image_file` or
        `read_many_image_files`.  Should contain validation or testing data (not
        training data), but this is not enforced.
    :param cnn_metadata_dict: Dictionary created by `train_cnn`.  This will
        ensure that data in `image_dict` are processed the exact same way as the
        training data for `cnn_model_object`.
    :param output_dir_name: Path to output directory.  Figures will be saved
        here.
    """

    predictor_matrix, _ = normalize_images(
        predictor_matrix=image_dict[PREDICTOR_MATRIX_KEY] + 0.,
        predictor_names=image_dict[PREDICTOR_NAMES_KEY],
        normalization_dict=cnn_metadata_dict[NORMALIZATION_DICT_KEY])
    predictor_matrix = predictor_matrix.astype('float32')

    target_values = binarize_target_images(
        target_matrix=image_dict[TARGET_MATRIX_KEY],
        binarization_threshold=cnn_metadata_dict[BINARIZATION_THRESHOLD_KEY])

    forecast_probabilities = _apply_cnn(cnn_model_object=cnn_model_object,
                                        predictor_matrix=predictor_matrix)
    print(MINOR_SEPARATOR_STRING)

    pofd_by_threshold, pod_by_threshold = roc_curves.plot_roc_curve(
        observed_labels=target_values,
        forecast_probabilities=forecast_probabilities)

    area_under_roc_curve = scikit_learn_auc(pofd_by_threshold, pod_by_threshold)
    title_string = 'Area under ROC curve: {0:.4f}'.format(area_under_roc_curve)

    pyplot.title(title_string)
    pyplot.show()

    _create_directory(directory_name=output_dir_name)
    roc_curve_file_name = '{0:s}/roc_curve.jpg'.format(output_dir_name)

    print('Saving figure to: "{0:s}"...'.format(roc_curve_file_name))
    pyplot.savefig(roc_curve_file_name, dpi=FIGURE_RESOLUTION_DPI)
    pyplot.close()

    performance_diagrams.plot_performance_diagram(
        observed_labels=target_values,
        forecast_probabilities=forecast_probabilities)
    pyplot.show()

    perf_diagram_file_name = '{0:s}/performance_diagram.jpg'.format(
        output_dir_name)

    print('Saving figure to: "{0:s}"...'.format(perf_diagram_file_name))
    pyplot.savefig(perf_diagram_file_name, dpi=FIGURE_RESOLUTION_DPI)
    pyplot.close()

    attributes_diagrams.plot_attributes_diagram(
        observed_labels=target_values,
        forecast_probabilities=forecast_probabilities, num_bins=20)
    pyplot.show()

    attr_diagram_file_name = '{0:s}/attributes_diagram.jpg'.format(
        output_dir_name)

    print('Saving figure to: "{0:s}"...'.format(attr_diagram_file_name))
    pyplot.savefig(attr_diagram_file_name, dpi=FIGURE_RESOLUTION_DPI)
    pyplot.close()


def evaluate_cnn_example(validation_image_dict):
    """Evaluates CNN on validation data.

    :param validation_image_dict: Dictionary created by `read_many_image_files`.
    """

    cnn_file_name = '{0:s}/pretrained_cnn/pretrained_cnn.h5'.format(
        MODULE4_DIR_NAME)
    cnn_metafile_name = find_model_metafile(model_file_name=cnn_file_name)

    cnn_model_object = read_keras_model(cnn_file_name)
    cnn_metadata_dict = read_model_metadata(cnn_metafile_name)
    validation_dir_name = '{0:s}/validation'.format(MODULE4_DIR_NAME)

    evaluate_cnn(
        cnn_model_object=cnn_model_object, image_dict=validation_image_dict,
        cnn_metadata_dict=cnn_metadata_dict,
        output_dir_name=validation_dir_name)
    print(SEPARATOR_STRING)


def bwo_example1(validation_image_dict, normalization_dict, cnn_model_object):
    """Optimizes random example (storm object) for positive class.

    :param validation_image_dict: Dictionary created by `read_many_image_files`.
    :param normalization_dict: Dictionary created by
        `get_image_normalization_params`.
    :param cnn_model_object: Trained instance of `keras.models.Model`.
    """

    orig_predictor_matrix = validation_image_dict[PREDICTOR_MATRIX_KEY][0, ...]
    predictor_names = validation_image_dict[PREDICTOR_NAMES_KEY]

    orig_predictor_matrix_norm, _ = normalize_images(
        predictor_matrix=orig_predictor_matrix + 0.,
        predictor_names=predictor_names, normalization_dict=normalization_dict)
    orig_predictor_matrix_norm = numpy.expand_dims(
        orig_predictor_matrix_norm, axis=0)

    optimized_predictor_matrix_norm = bwo_for_class(
        cnn_model_object=cnn_model_object, target_class=1,
        init_function_or_matrices=[orig_predictor_matrix_norm]
    )[0][0, ...]

    optimized_predictor_matrix = denormalize_images(
        predictor_matrix=optimized_predictor_matrix_norm,
        predictor_names=predictor_names, normalization_dict=normalization_dict)

    temperature_index = predictor_names.index(TEMPERATURE_NAME)
    combined_temp_matrix_kelvins = numpy.concatenate(
        (orig_predictor_matrix[..., temperature_index],
         optimized_predictor_matrix[..., temperature_index]),
        axis=0)

    min_colour_temp_kelvins = numpy.percentile(combined_temp_matrix_kelvins, 1)
    max_colour_temp_kelvins = numpy.percentile(combined_temp_matrix_kelvins, 99)

    figure_object, _ = plot_many_predictors_with_barbs(
        predictor_matrix=orig_predictor_matrix,
        predictor_names=predictor_names,
        min_colour_temp_kelvins=min_colour_temp_kelvins,
        max_colour_temp_kelvins=max_colour_temp_kelvins)

    figure_object.suptitle('Real example (before optimization)')
    pyplot.show()

    figure_object, _ = plot_many_predictors_with_barbs(
        predictor_matrix=optimized_predictor_matrix,
        predictor_names=predictor_names,
        min_colour_temp_kelvins=min_colour_temp_kelvins,
        max_colour_temp_kelvins=max_colour_temp_kelvins)

    figure_object.suptitle('Synthetic example (after optimization)')
    pyplot.show()


def bwo_example2(validation_image_dict, normalization_dict, cnn_model_object):
    """Optimizes random example (storm object) for negative class.

    :param validation_image_dict: Dictionary created by `read_many_image_files`.
    :param normalization_dict: Dictionary created by
        `get_image_normalization_params`.
    :param cnn_model_object: Trained instance of `keras.models.Model`.
    """

    orig_predictor_matrix = validation_image_dict[PREDICTOR_MATRIX_KEY][0, ...]
    predictor_names = validation_image_dict[PREDICTOR_NAMES_KEY]

    orig_predictor_matrix_norm, _ = normalize_images(
        predictor_matrix=orig_predictor_matrix + 0.,
        predictor_names=predictor_names, normalization_dict=normalization_dict)
    orig_predictor_matrix_norm = numpy.expand_dims(
        orig_predictor_matrix_norm, axis=0)

    optimized_predictor_matrix_norm = bwo_for_class(
        cnn_model_object=cnn_model_object, target_class=0,
        init_function_or_matrices=[orig_predictor_matrix_norm]
    )[0][0, ...]

    optimized_predictor_matrix = denormalize_images(
        predictor_matrix=optimized_predictor_matrix_norm,
        predictor_names=predictor_names, normalization_dict=normalization_dict)

    temperature_index = predictor_names.index(TEMPERATURE_NAME)
    combined_temp_matrix_kelvins = numpy.concatenate(
        (orig_predictor_matrix[..., temperature_index],
         optimized_predictor_matrix[..., temperature_index]),
        axis=0)

    min_colour_temp_kelvins = numpy.percentile(combined_temp_matrix_kelvins, 1)
    max_colour_temp_kelvins = numpy.percentile(combined_temp_matrix_kelvins, 99)

    figure_object, _ = plot_many_predictors_with_barbs(
        predictor_matrix=orig_predictor_matrix,
        predictor_names=predictor_names,
        min_colour_temp_kelvins=min_colour_temp_kelvins,
        max_colour_temp_kelvins=max_colour_temp_kelvins)

    figure_object.suptitle('Real example (before optimization)')
    pyplot.show()

    figure_object, _ = plot_many_predictors_with_barbs(
        predictor_matrix=optimized_predictor_matrix,
        predictor_names=predictor_names,
        min_colour_temp_kelvins=min_colour_temp_kelvins,
        max_colour_temp_kelvins=max_colour_temp_kelvins)

    figure_object.suptitle('Synthetic example (after optimization)')
    pyplot.show()


def bwo_example3(validation_image_dict, normalization_dict, cnn_model_object):
    """Optimizes extreme example (storm object) for positive class.

    :param validation_image_dict: Dictionary created by `read_many_image_files`.
    :param normalization_dict: Dictionary created by
        `get_image_normalization_params`.
    :param cnn_model_object: Trained instance of `keras.models.Model`.
    """

    target_matrix_s01 = validation_image_dict[TARGET_MATRIX_KEY]
    example_index = numpy.unravel_index(
        numpy.argmax(target_matrix_s01), target_matrix_s01.shape
    )[0]

    orig_predictor_matrix = validation_image_dict[PREDICTOR_MATRIX_KEY][
        example_index, ...]
    predictor_names = validation_image_dict[PREDICTOR_NAMES_KEY]

    orig_predictor_matrix_norm, _ = normalize_images(
        predictor_matrix=orig_predictor_matrix + 0.,
        predictor_names=predictor_names, normalization_dict=normalization_dict)
    orig_predictor_matrix_norm = numpy.expand_dims(
        orig_predictor_matrix_norm, axis=0)

    optimized_predictor_matrix_norm = bwo_for_class(
        cnn_model_object=cnn_model_object, target_class=1,
        init_function_or_matrices=[orig_predictor_matrix_norm]
    )[0][0, ...]

    optimized_predictor_matrix = denormalize_images(
        predictor_matrix=optimized_predictor_matrix_norm,
        predictor_names=predictor_names, normalization_dict=normalization_dict)

    temperature_index = predictor_names.index(TEMPERATURE_NAME)
    combined_temp_matrix_kelvins = numpy.concatenate(
        (orig_predictor_matrix[..., temperature_index],
         optimized_predictor_matrix[..., temperature_index]),
        axis=0)

    min_colour_temp_kelvins = numpy.percentile(combined_temp_matrix_kelvins, 1)
    max_colour_temp_kelvins = numpy.percentile(combined_temp_matrix_kelvins, 99)

    figure_object, _ = plot_many_predictors_with_barbs(
        predictor_matrix=orig_predictor_matrix,
        predictor_names=predictor_names,
        min_colour_temp_kelvins=min_colour_temp_kelvins,
        max_colour_temp_kelvins=max_colour_temp_kelvins)

    figure_object.suptitle('Real example (before optimization)')
    pyplot.show()

    figure_object, _ = plot_many_predictors_with_barbs(
        predictor_matrix=optimized_predictor_matrix,
        predictor_names=predictor_names,
        min_colour_temp_kelvins=min_colour_temp_kelvins,
        max_colour_temp_kelvins=max_colour_temp_kelvins)

    figure_object.suptitle('Synthetic example (after optimization)')
    pyplot.show()


def bwo_example4(validation_image_dict, normalization_dict, cnn_model_object):
    """Optimizes extreme example (storm object) for negative class.

    :param validation_image_dict: Dictionary created by `read_many_image_files`.
    :param normalization_dict: Dictionary created by
        `get_image_normalization_params`.
    :param cnn_model_object: Trained instance of `keras.models.Model`.
    """

    target_matrix_s01 = validation_image_dict[TARGET_MATRIX_KEY]
    example_index = numpy.unravel_index(
        numpy.argmax(target_matrix_s01), target_matrix_s01.shape
    )[0]

    orig_predictor_matrix = validation_image_dict[PREDICTOR_MATRIX_KEY][
        example_index, ...]
    predictor_names = validation_image_dict[PREDICTOR_NAMES_KEY]

    orig_predictor_matrix_norm, _ = normalize_images(
        predictor_matrix=orig_predictor_matrix + 0.,
        predictor_names=predictor_names, normalization_dict=normalization_dict)
    orig_predictor_matrix_norm = numpy.expand_dims(
        orig_predictor_matrix_norm, axis=0)

    optimized_predictor_matrix_norm = bwo_for_class(
        cnn_model_object=cnn_model_object, target_class=0,
        init_function_or_matrices=[orig_predictor_matrix_norm]
    )[0][0, ...]

    optimized_predictor_matrix = denormalize_images(
        predictor_matrix=optimized_predictor_matrix_norm,
        predictor_names=predictor_names, normalization_dict=normalization_dict)

    temperature_index = predictor_names.index(TEMPERATURE_NAME)
    combined_temp_matrix_kelvins = numpy.concatenate(
        (orig_predictor_matrix[..., temperature_index],
         optimized_predictor_matrix[..., temperature_index]),
        axis=0)

    min_colour_temp_kelvins = numpy.percentile(combined_temp_matrix_kelvins, 1)
    max_colour_temp_kelvins = numpy.percentile(combined_temp_matrix_kelvins, 99)

    figure_object, _ = plot_many_predictors_with_barbs(
        predictor_matrix=orig_predictor_matrix,
        predictor_names=predictor_names,
        min_colour_temp_kelvins=min_colour_temp_kelvins,
        max_colour_temp_kelvins=max_colour_temp_kelvins)

    figure_object.suptitle('Real example (before optimization)')
    pyplot.show()

    figure_object, _ = plot_many_predictors_with_barbs(
        predictor_matrix=optimized_predictor_matrix,
        predictor_names=predictor_names,
        min_colour_temp_kelvins=min_colour_temp_kelvins,
        max_colour_temp_kelvins=max_colour_temp_kelvins)

    figure_object.suptitle('Synthetic example (after optimization)')
    pyplot.show()


def _create_smoothing_filter(
        smoothing_radius_px, num_half_filter_rows, num_half_filter_columns,
        num_channels):
    """Creates convolution filter for Gaussian smoothing.

    M = number of rows in filter
    N = number of columns in filter
    C = number of channels (or "variables" or "features") to smooth.  Each
        channel will be smoothed independently.

    :param smoothing_radius_px: e-folding radius (pixels).
    :param num_half_filter_rows: Number of rows in one half of filter.  Total
        number of rows will be 2 * `num_half_filter_rows` + 1.
    :param num_half_filter_columns: Same but for columns.
    :param num_channels: C in the above discussion.
    :return: weight_matrix: M-by-N-by-C-by-C numpy array of convolution weights.
    """

    num_filter_rows = 2 * num_half_filter_rows + 1
    num_filter_columns = 2 * num_half_filter_columns + 1

    row_offsets_unique = numpy.linspace(
        -num_half_filter_rows, num_half_filter_rows, num=num_filter_rows,
        dtype=float)
    column_offsets_unique = numpy.linspace(
        -num_half_filter_columns, num_half_filter_columns,
        num=num_filter_columns, dtype=float)

    column_offset_matrix, row_offset_matrix = numpy.meshgrid(
        column_offsets_unique, row_offsets_unique)

    pixel_offset_matrix = numpy.sqrt(
        row_offset_matrix ** 2 + column_offset_matrix ** 2)

    small_weight_matrix = numpy.exp(
        -pixel_offset_matrix ** 2 / (2 * smoothing_radius_px ** 2)
    )
    small_weight_matrix = small_weight_matrix / numpy.sum(small_weight_matrix)

    weight_matrix = numpy.zeros(
        (num_filter_rows, num_filter_columns, num_channels, num_channels)
    )

    for k in range(num_channels):
        weight_matrix[..., k, k] = small_weight_matrix

    return weight_matrix


def setup_ucn(
        num_input_features, first_num_rows, first_num_columns,
        upsampling_factors, num_output_channels,
        use_activation_for_out_layer=False, use_bn_for_out_layer=True,
        use_transposed_conv=False, smoothing_radius_px=None):
    """Creates (but does not train) upconvnet.

    L = number of conv or deconv layers

    :param num_input_features: Number of input features.
    :param first_num_rows: Number of rows in input to first deconv layer.  The
        input features will be reshaped into a grid with this many rows.
    :param first_num_columns: Same but for columns.
    :param upsampling_factors: length-L numpy array of upsampling factors.  Must
        all be positive integers.
    :param num_output_channels: Number of channels in output images.
    :param use_activation_for_out_layer: Boolean flag.  If True, activation will
        be applied to output layer.
    :param use_bn_for_out_layer: Boolean flag.  If True, batch normalization
        will be applied to output layer.
    :param use_transposed_conv: Boolean flag.  If True, upsampling will be done
        with transposed-convolution layers.  If False, each upsampling will be
        done with an upsampling layer followed by a conv layer.
    :param smoothing_radius_px: Smoothing radius (pixels).  Gaussian smoothing
        with this e-folding radius will be done after each upsampling.  If
        `smoothing_radius_px is None`, no smoothing will be done.
    :return: ucn_model_object: Untrained instance of `keras.models.Model`.
    """

    if smoothing_radius_px is not None:
        num_half_smoothing_rows = int(numpy.round(
            (NUM_SMOOTHING_FILTER_ROWS - 1) / 2
        ))
        num_half_smoothing_columns = int(numpy.round(
            (NUM_SMOOTHING_FILTER_COLUMNS - 1) / 2
        ))

    regularizer_object = keras.regularizers.l1_l2(l1=L1_WEIGHT, l2=L2_WEIGHT)
    input_layer_object = keras.layers.Input(shape=(num_input_features,))

    current_num_filters = int(numpy.round(
        num_input_features / (first_num_rows * first_num_columns)
    ))

    layer_object = keras.layers.Reshape(
        target_shape=(first_num_rows, first_num_columns, current_num_filters)
    )(input_layer_object)

    num_main_layers = len(upsampling_factors)

    for i in range(num_main_layers):
        this_upsampling_factor = upsampling_factors[i]

        if i == num_main_layers - 1:
            current_num_filters = num_output_channels + 0
        elif this_upsampling_factor == 1:
            current_num_filters = int(numpy.round(current_num_filters / 2))

        if use_transposed_conv:
            if this_upsampling_factor > 1:
                this_padding_arg = 'same'
            else:
                this_padding_arg = 'valid'

            layer_object = keras.layers.Conv2DTranspose(
                filters=current_num_filters,
                kernel_size=(NUM_CONV_FILTER_ROWS, NUM_CONV_FILTER_COLUMNS),
                strides=(this_upsampling_factor, this_upsampling_factor),
                padding=this_padding_arg, data_format='channels_last',
                dilation_rate=(1, 1), activation=None, use_bias=True,
                kernel_initializer='glorot_uniform', bias_initializer='zeros',
                kernel_regularizer=regularizer_object
            )(layer_object)

        else:
            if this_upsampling_factor > 1:
                try:
                    layer_object = keras.layers.UpSampling2D(
                        size=(this_upsampling_factor, this_upsampling_factor),
                        data_format='channels_last', interpolation='nearest'
                    )(layer_object)
                except:
                    layer_object = keras.layers.UpSampling2D(
                        size=(this_upsampling_factor, this_upsampling_factor),
                        data_format='channels_last'
                    )(layer_object)

            layer_object = keras.layers.Conv2D(
                filters=current_num_filters,
                kernel_size=(NUM_CONV_FILTER_ROWS, NUM_CONV_FILTER_COLUMNS),
                strides=(1, 1), padding='same', data_format='channels_last',
                dilation_rate=(1, 1), activation=None, use_bias=True,
                kernel_initializer='glorot_uniform', bias_initializer='zeros',
                kernel_regularizer=regularizer_object
            )(layer_object)

            if this_upsampling_factor == 1:
                layer_object = keras.layers.ZeroPadding2D(
                    padding=(1, 1), data_format='channels_last'
                )(layer_object)

        if smoothing_radius_px is not None:
            this_weight_matrix = _create_smoothing_filter(
                smoothing_radius_px=smoothing_radius_px,
                num_half_filter_rows=num_half_smoothing_rows,
                num_half_filter_columns=num_half_smoothing_columns,
                num_channels=current_num_filters)

            this_bias_vector = numpy.zeros(current_num_filters)

            layer_object = keras.layers.Conv2D(
                filters=current_num_filters,
                kernel_size=(NUM_SMOOTHING_FILTER_ROWS,
                             NUM_SMOOTHING_FILTER_COLUMNS),
                strides=(1, 1), padding='same', data_format='channels_last',
                dilation_rate=(1, 1), activation=None, use_bias=True,
                kernel_initializer='glorot_uniform', bias_initializer='zeros',
                kernel_regularizer=regularizer_object, trainable=False,
                weights=[this_weight_matrix, this_bias_vector]
            )(layer_object)

        if i < num_main_layers - 1 or use_activation_for_out_layer:
            layer_object = keras.layers.LeakyReLU(
                alpha=SLOPE_FOR_RELU
            )(layer_object)

        if i < num_main_layers - 1 or use_bn_for_out_layer:
            layer_object = keras.layers.BatchNormalization(
                axis=-1, center=True, scale=True
            )(layer_object)

    ucn_model_object = keras.models.Model(
        inputs=input_layer_object, outputs=layer_object)
    ucn_model_object.compile(
        loss=keras.losses.mean_squared_error, optimizer=keras.optimizers.Adam())

    ucn_model_object.summary()
    return ucn_model_object


def get_cnn_flatten_layer(cnn_model_object):
    """Finds flattening layer in CNN.

    This method assumes that there is only one flattening layer.  If there are
    several, this method will return the first (shallowest).

    :param cnn_model_object: Instance of `keras.models.Model`.
    :return: layer_name: Name of flattening layer.
    :raises: TypeError: if flattening layer cannot be found.
    """

    layer_names = [lyr.name for lyr in cnn_model_object.layers]

    flattening_flags = numpy.array(
        ['flatten' in n for n in layer_names], dtype=bool)
    flattening_indices = numpy.where(flattening_flags)[0]

    if len(flattening_indices) == 0:
        error_string = (
            'Cannot find flattening layer in model.  Layer names are listed '
            'below.\n{0:s}'
        ).format(str(layer_names))

        raise TypeError(error_string)

    return layer_names[flattening_indices[0]]


def setup_ucn_example(cnn_model_object):
    """Example of UCN architecture (with transposed conv, no smoothing).

    :param cnn_model_object: Trained CNN (instance of `keras.models.Model`).
    """

    cnn_feature_layer_name = get_cnn_flatten_layer(cnn_model_object)
    cnn_feature_layer_object = cnn_model_object.get_layer(
        name=cnn_feature_layer_name)
    cnn_feature_dimensions = numpy.array(
        cnn_feature_layer_object.input.shape[1:], dtype=int)

    num_input_features = numpy.prod(cnn_feature_dimensions)
    first_num_rows = cnn_feature_dimensions[0]
    first_num_columns = cnn_feature_dimensions[1]
    num_output_channels = numpy.array(
        cnn_model_object.input.shape[1:], dtype=int
    )[-1]

    upsampling_factors = numpy.array([2, 1, 1, 2, 1, 1], dtype=int)

    ucn_model_object = setup_ucn(
        num_input_features=num_input_features, first_num_rows=first_num_rows,
        first_num_columns=first_num_columns,
        upsampling_factors=upsampling_factors,
        num_output_channels=num_output_channels,
        use_transposed_conv=True, smoothing_radius_px=None)


def ucn_generator(netcdf_file_names, num_examples_per_batch, normalization_dict,
                  cnn_model_object, cnn_feature_layer_name):
    """Generates training examples for UCN (upconvolutional network) on the fly.

    E = number of examples (storm objects)
    M = number of rows in each storm-centered grid
    N = number of columns in each storm-centered grid
    C = number of channels (predictor variables)
    Z = number of scalar features (neurons in layer `cnn_feature_layer_name` of
        the CNN specified by `cnn_model_object`)

    :param netcdf_file_names: 1-D list of paths to input (NetCDF) files.
    :param num_examples_per_batch: Number of examples per training batch.
    :param normalization_dict: See doc for `normalize_images`.  You cannot leave
        this as None.
    :param cnn_model_object: Trained CNN model (instance of
        `keras.models.Model`).  This will be used to turn images stored in
        `netcdf_file_names` into scalar features.
    :param cnn_feature_layer_name: The "scalar features" will be the set of
        activations from this layer.
    :return: feature_matrix: E-by-Z numpy array of scalar features.  These are
        the "predictors" for the upconv network.
    :return: target_matrix: E-by-M-by-N-by-C numpy array of target images.
        These are the predictors for the CNN and the targets for the upconv
        network.
    :raises: TypeError: if `normalization_dict is None`.
    """

    if normalization_dict is None:
        error_string = 'normalization_dict cannot be None.  Must be specified.'
        raise TypeError(error_string)

    random.shuffle(netcdf_file_names)
    num_files = len(netcdf_file_names)
    file_index = 0

    num_examples_in_memory = 0
    full_target_matrix = None
    predictor_names = None

    while True:
        while num_examples_in_memory < num_examples_per_batch:
            print('Reading data from: "{0:s}"...'.format(
                netcdf_file_names[file_index]))

            this_image_dict = read_image_file(netcdf_file_names[file_index])
            predictor_names = this_image_dict[PREDICTOR_NAMES_KEY]

            file_index += 1
            if file_index >= num_files:
                file_index = 0

            if full_target_matrix is None or full_target_matrix.size == 0:
                full_target_matrix = this_image_dict[PREDICTOR_MATRIX_KEY] + 0.
            else:
                full_target_matrix = numpy.concatenate(
                    (full_target_matrix, this_image_dict[PREDICTOR_MATRIX_KEY]),
                    axis=0)

            num_examples_in_memory = full_target_matrix.shape[0]

        batch_indices = numpy.linspace(
            0, num_examples_in_memory - 1, num=num_examples_in_memory,
            dtype=int)
        batch_indices = numpy.random.choice(
            batch_indices, size=num_examples_per_batch, replace=False)

        target_matrix, _ = normalize_images(
            predictor_matrix=full_target_matrix[batch_indices, ...],
            predictor_names=predictor_names,
            normalization_dict=normalization_dict)
        target_matrix = target_matrix.astype('float32')

        feature_matrix = _apply_cnn(
            cnn_model_object=cnn_model_object, predictor_matrix=target_matrix,
            verbose=False, output_layer_name=cnn_feature_layer_name)

        num_examples_in_memory = 0
        full_target_matrix = None

        yield (feature_matrix, target_matrix)


def train_ucn(
        ucn_model_object, training_file_names, normalization_dict,
        cnn_model_object, cnn_file_name, cnn_feature_layer_name,
        num_examples_per_batch, num_epochs, num_training_batches_per_epoch,
        output_model_file_name, validation_file_names=None,
        num_validation_batches_per_epoch=None):
    """Trains UCN (upconvolutional network).

    :param ucn_model_object: Untrained instance of `keras.models.Model` (may be
        created by `setup_ucn`), representing the upconv network.
    :param training_file_names: 1-D list of paths to training files (must be
        readable by `read_image_file`).
    :param normalization_dict: See doc for `ucn_generator`.
    :param cnn_model_object: Same.
    :param cnn_file_name: Path to file with trained CNN (represented by
        `cnn_model_object`).  This is needed only for the output dictionary
        (metadata).
    :param cnn_feature_layer_name: Same.
    :param num_examples_per_batch: Same.
    :param num_epochs: Number of epochs.
    :param num_training_batches_per_epoch: Number of training batches furnished
        to model in each epoch.
    :param output_model_file_name: Path to output file.  The model will be saved
        as an HDF5 file (extension should be ".h5", but this is not enforced).
    :param validation_file_names: 1-D list of paths to training files (must be
        readable by `read_image_file`).  If `validation_file_names is None`,
        will omit on-the-fly validation.
    :param num_validation_batches_per_epoch:
        [used only if `validation_file_names is not None`]
        Number of validation batches furnished to model in each epoch.

    :return: ucn_metadata_dict: Dictionary with the following keys.
    ucn_metadata_dict['training_file_names']: See input doc.
    ucn_metadata_dict['normalization_dict']: Same.
    ucn_metadata_dict['cnn_file_name']: Same.
    ucn_metadata_dict['cnn_feature_layer_name']: Same.
    ucn_metadata_dict['num_examples_per_batch']: Same.
    ucn_metadata_dict['num_training_batches_per_epoch']: Same.
    ucn_metadata_dict['validation_file_names']: Same.
    ucn_metadata_dict['num_validation_batches_per_epoch']: Same.
    """

    _create_directory(file_name=output_model_file_name)

    if validation_file_names is None:
        checkpoint_object = keras.callbacks.ModelCheckpoint(
            filepath=output_model_file_name, monitor='loss', verbose=1,
            save_best_only=False, save_weights_only=False, mode='min',
            period=1)
    else:
        checkpoint_object = keras.callbacks.ModelCheckpoint(
            filepath=output_model_file_name, monitor='val_loss', verbose=1,
            save_best_only=True, save_weights_only=False, mode='min',
            period=1)

    list_of_callback_objects = [checkpoint_object]

    ucn_metadata_dict = {
        TRAINING_FILES_KEY: training_file_names,
        NORMALIZATION_DICT_KEY: normalization_dict,
        CNN_FILE_KEY: cnn_file_name,
        CNN_FEATURE_LAYER_KEY: cnn_feature_layer_name,
        NUM_EXAMPLES_PER_BATCH_KEY: num_examples_per_batch,
        NUM_TRAINING_BATCHES_KEY: num_training_batches_per_epoch,
        VALIDATION_FILES_KEY: validation_file_names,
        NUM_VALIDATION_BATCHES_KEY: num_validation_batches_per_epoch
    }

    training_generator = ucn_generator(
        netcdf_file_names=training_file_names,
        num_examples_per_batch=num_examples_per_batch,
        normalization_dict=normalization_dict,
        cnn_model_object=cnn_model_object,
        cnn_feature_layer_name=cnn_feature_layer_name)

    if validation_file_names is None:
        ucn_model_object.fit_generator(
            generator=training_generator,
            steps_per_epoch=num_training_batches_per_epoch, epochs=num_epochs,
            verbose=1, callbacks=list_of_callback_objects, workers=0)

        return ucn_metadata_dict

    early_stopping_object = keras.callbacks.EarlyStopping(
        monitor='val_loss', min_delta=MIN_MSE_DECREASE_FOR_EARLY_STOP,
        patience=NUM_EPOCHS_FOR_EARLY_STOPPING, verbose=1, mode='min')

    list_of_callback_objects.append(early_stopping_object)

    validation_generator = ucn_generator(
        netcdf_file_names=validation_file_names,
        num_examples_per_batch=num_examples_per_batch,
        normalization_dict=normalization_dict,
        cnn_model_object=cnn_model_object,
        cnn_feature_layer_name=cnn_feature_layer_name)

    ucn_model_object.fit_generator(
        generator=training_generator,
        steps_per_epoch=num_training_batches_per_epoch, epochs=num_epochs,
        verbose=1, callbacks=list_of_callback_objects, workers=0,
        validation_data=validation_generator,
        validation_steps=num_validation_batches_per_epoch)

    return ucn_metadata_dict


def train_ucn_example(ucn_model_object, training_file_names, normalization_dict,
                      cnn_model_object, cnn_file_name):
    """Actually trains the UCN (upconvolutional network).

    :param ucn_model_object: See doc for `train_ucn`.
    :param training_file_names: Same.
    :param normalization_dict: Same.
    :param cnn_model_object: See doc for `cnn_model_object` in `train_ucn`.
    :param cnn_file_name: See doc for `train_ucn`.
    """

    validation_file_names = find_many_image_files(
        first_date_string='20150101', last_date_string='20151231')

    ucn_file_name = '{0:s}/ucn_model.h5'.format(MODULE4_DIR_NAME)
    ucn_metadata_dict = train_ucn(
        ucn_model_object=ucn_model_object,
        training_file_names=training_file_names,
        normalization_dict=normalization_dict,
        cnn_model_object=cnn_model_object, cnn_file_name=cnn_file_name,
        cnn_feature_layer_name=get_cnn_flatten_layer(cnn_model_object),
        num_examples_per_batch=100, num_epochs=10,
        num_training_batches_per_epoch=10, output_model_file_name=ucn_file_name,
        validation_file_names=validation_file_names,
        num_validation_batches_per_epoch=10)


def apply_ucn_example1(
        validation_image_dict, normalization_dict, cnn_model_object):
    """Uses upconvnet to reconstruct random validation example.

    :param validation_image_dict: Dictionary created by `read_many_image_files`.
    :param normalization_dict: Dictionary created by
        `get_image_normalization_params`.
    :param cnn_model_object: Trained instance of `keras.models.Model`,
        representing the CNN that goes with the upconvnet.
    """

    ucn_file_name = '{0:s}/pretrained_cnn/pretrained_ucn.h5'.format(
        MODULE4_DIR_NAME)
    ucn_metafile_name = find_model_metafile(model_file_name=ucn_file_name)

    ucn_model_object = read_keras_model(ucn_file_name)
    ucn_metadata_dict = read_model_metadata(ucn_metafile_name)

    image_matrix = validation_image_dict[PREDICTOR_MATRIX_KEY][0, ...]
    predictor_names = validation_image_dict[PREDICTOR_NAMES_KEY]

    image_matrix_norm, _ = normalize_images(
        predictor_matrix=image_matrix + 0.,
        predictor_names=predictor_names, normalization_dict=normalization_dict)
    image_matrix_norm = numpy.expand_dims(image_matrix_norm, axis=0)

    feature_matrix = _apply_cnn(
        cnn_model_object=cnn_model_object, predictor_matrix=image_matrix_norm,
        output_layer_name=get_cnn_flatten_layer(cnn_model_object),
        verbose=False)

    reconstructed_image_matrix_norm = ucn_model_object.predict(
        feature_matrix, batch_size=1)

    reconstructed_image_matrix = denormalize_images(
        predictor_matrix=reconstructed_image_matrix_norm,
        predictor_names=predictor_names, normalization_dict=normalization_dict
    )[0, ...]

    temperature_index = predictor_names.index(TEMPERATURE_NAME)
    combined_temp_matrix_kelvins = numpy.concatenate(
        (image_matrix[..., temperature_index],
         reconstructed_image_matrix[..., temperature_index]),
        axis=0)

    min_colour_temp_kelvins = numpy.percentile(combined_temp_matrix_kelvins, 1)
    max_colour_temp_kelvins = numpy.percentile(combined_temp_matrix_kelvins, 99)

    figure_object, _ = plot_many_predictors_with_barbs(
        predictor_matrix=image_matrix,
        predictor_names=predictor_names,
        min_colour_temp_kelvins=min_colour_temp_kelvins,
        max_colour_temp_kelvins=max_colour_temp_kelvins)

    figure_object.suptitle('Original image (CNN input)')
    pyplot.show()

    figure_object, _ = plot_many_predictors_with_barbs(
        predictor_matrix=reconstructed_image_matrix,
        predictor_names=predictor_names,
        min_colour_temp_kelvins=min_colour_temp_kelvins,
        max_colour_temp_kelvins=max_colour_temp_kelvins)

    figure_object.suptitle('Reconstructed image (upconvnet output)')
    pyplot.show()


def apply_ucn_example2(
        validation_image_dict, normalization_dict, ucn_model_object,
        cnn_model_object):
    """Uses upconvnet to reconstruct extreme validation example.

    :param validation_image_dict: Dictionary created by `read_many_image_files`.
    :param normalization_dict: Dictionary created by
        `get_image_normalization_params`.
    :param ucn_model_object: Trained instance of `keras.models.Model`,
        representing the upconvnet.
    :param cnn_model_object: Trained instance of `keras.models.Model`,
        representing the CNN that goes with the upconvnet.
    """

    target_matrix_s01 = validation_image_dict[TARGET_MATRIX_KEY]
    example_index = numpy.unravel_index(
        numpy.argmax(target_matrix_s01), target_matrix_s01.shape
    )[0]

    image_matrix = validation_image_dict[PREDICTOR_MATRIX_KEY][
        example_index, ...]
    predictor_names = validation_image_dict[PREDICTOR_NAMES_KEY]

    image_matrix_norm, _ = normalize_images(
        predictor_matrix=image_matrix + 0.,
        predictor_names=predictor_names, normalization_dict=normalization_dict)
    image_matrix_norm = numpy.expand_dims(image_matrix_norm, axis=0)

    feature_matrix = _apply_cnn(
        cnn_model_object=cnn_model_object, predictor_matrix=image_matrix_norm,
        output_layer_name=get_cnn_flatten_layer(cnn_model_object),
        verbose=False)

    reconstructed_image_matrix_norm = ucn_model_object.predict(
        feature_matrix, batch_size=1)

    reconstructed_image_matrix = denormalize_images(
        predictor_matrix=reconstructed_image_matrix_norm,
        predictor_names=predictor_names, normalization_dict=normalization_dict
    )[0, ...]

    temperature_index = predictor_names.index(TEMPERATURE_NAME)
    combined_temp_matrix_kelvins = numpy.concatenate(
        (image_matrix[..., temperature_index],
         reconstructed_image_matrix[..., temperature_index]),
        axis=0)

    min_colour_temp_kelvins = numpy.percentile(combined_temp_matrix_kelvins, 1)
    max_colour_temp_kelvins = numpy.percentile(combined_temp_matrix_kelvins, 99)

    figure_object, _ = plot_many_predictors_with_barbs(
        predictor_matrix=image_matrix,
        predictor_names=predictor_names,
        min_colour_temp_kelvins=min_colour_temp_kelvins,
        max_colour_temp_kelvins=max_colour_temp_kelvins)

    figure_object.suptitle('Original image (CNN input)')
    pyplot.show()

    figure_object, _ = plot_many_predictors_with_barbs(
        predictor_matrix=reconstructed_image_matrix,
        predictor_names=predictor_names,
        min_colour_temp_kelvins=min_colour_temp_kelvins,
        max_colour_temp_kelvins=max_colour_temp_kelvins)

    figure_object.suptitle('Reconstructed image (upconvnet output)')
    pyplot.show()


def _normalize_features(feature_matrix, feature_means=None,
                        feature_standard_deviations=None):
    """Normalizes scalar features to z-scores.

    E = number of examples (storm objects)
    Z = number of features

    :param feature_matrix: E-by-Z numpy array of features.
    :param feature_means: length-Z numpy array of mean values.  If
        `feature_means is None`, these will be computed on the fly from
        `feature_matrix`.
    :param feature_standard_deviations: Same but with standard deviations.
    :return: feature_matrix: Normalized version of input.
    :return: feature_means: See input doc.
    :return: feature_standard_deviations: See input doc.
    """

    if feature_means is None or feature_standard_deviations is None:
        feature_means = numpy.mean(feature_matrix, axis=0)
        feature_standard_deviations = numpy.std(feature_matrix, axis=0, ddof=1)

    num_examples = feature_matrix.shape[0]
    num_features = feature_matrix.shape[1]

    mean_matrix = numpy.reshape(feature_means, (1, num_features))
    mean_matrix = numpy.repeat(mean_matrix, repeats=num_examples, axis=0)

    stdev_matrix = numpy.reshape(feature_standard_deviations, (1, num_features))
    stdev_matrix = numpy.repeat(stdev_matrix, repeats=num_examples, axis=0)

    feature_matrix = (feature_matrix - mean_matrix) / stdev_matrix
    return feature_matrix, feature_means, feature_standard_deviations


def _fit_svd(baseline_feature_matrix, test_feature_matrix,
             percent_variance_to_keep):
    """Fits SVD (singular-value decomposition) model.

    B = number of baseline examples (storm objects)
    T = number of testing examples (storm objects)
    Z = number of scalar features (produced by dense layer of a CNN)
    K = number of modes (top eigenvectors) retained

    The SVD model will be fit only to the baseline set, but both the baseline
    and testing sets will be used to compute normalization parameters (means and
    standard deviations).  Before, when only the baseline set was used to
    compute normalization params, the testing set had huge standard deviations,
    which caused the results of novelty detection to be physically unrealistic.

    :param baseline_feature_matrix: B-by-Z numpy array of features.
    :param test_feature_matrix: T-by-Z numpy array of features.
    :param percent_variance_to_keep: Percentage of variance to keep.  Determines
        how many eigenvectors (K in the above discussion) will be used in the
        SVD model.
    :return: svd_dictionary: Dictionary with the following keys.
    svd_dictionary['eof_matrix']: Z-by-K numpy array, where each column is an
        EOF (empirical orthogonal function).
    svd_dictionary['feature_means']: length-Z numpy array with mean value of
        each feature (before transformation).
    svd_dictionary['feature_standard_deviations']: length-Z numpy array with
        standard deviation of each feature (before transformation).
    """

    combined_feature_matrix = numpy.concatenate(
        (baseline_feature_matrix, test_feature_matrix), axis=0)

    combined_feature_matrix, feature_means, feature_standard_deviations = (
        _normalize_features(feature_matrix=combined_feature_matrix)
    )

    num_baseline_examples = baseline_feature_matrix.shape[0]
    baseline_feature_matrix = combined_feature_matrix[
        :num_baseline_examples, ...]

    eigenvalues, eof_matrix = numpy.linalg.svd(baseline_feature_matrix)[1:]
    eigenvalues = eigenvalues ** 2

    explained_variances = eigenvalues / numpy.sum(eigenvalues)
    cumulative_explained_variances = numpy.cumsum(explained_variances)

    fraction_of_variance_to_keep = 0.01 * percent_variance_to_keep
    num_modes_to_keep = 1 + numpy.where(
        cumulative_explained_variances >= fraction_of_variance_to_keep
    )[0][0]

    print(
        ('Number of modes required to explain {0:f}% of variance: {1:d}'
         ).format(percent_variance_to_keep, num_modes_to_keep)
    )

    return {
        EOF_MATRIX_KEY: numpy.transpose(eof_matrix)[..., :num_modes_to_keep],
        FEATURE_MEANS_KEY: feature_means,
        FEATURE_STDEVS_KEY: feature_standard_deviations
    }


def _apply_svd(feature_vector, svd_dictionary):
    """Applies SVD (singular-value decomposition) model to new example.

    Z = number of features

    :param feature_vector: length-Z numpy array with feature values for one
        example (storm object).
    :param svd_dictionary: Dictionary created by `_fit_svd`.
    :return: reconstructed_feature_vector: Reconstructed version of input.
    """

    this_matrix = numpy.dot(
        svd_dictionary[EOF_MATRIX_KEY],
        numpy.transpose(svd_dictionary[EOF_MATRIX_KEY])
    )

    feature_vector_norm = (
        (feature_vector - svd_dictionary[FEATURE_MEANS_KEY]) /
        svd_dictionary[FEATURE_STDEVS_KEY]
    )

    reconstructed_feature_vector_norm = numpy.dot(
        this_matrix, feature_vector_norm)

    return (
        svd_dictionary[FEATURE_MEANS_KEY] +
        reconstructed_feature_vector_norm * svd_dictionary[FEATURE_STDEVS_KEY]
    )


def do_novelty_detection(
        baseline_image_matrix, test_image_matrix, image_normalization_dict,
        predictor_names, cnn_model_object, cnn_feature_layer_name,
        ucn_model_object, num_novel_test_images,
        percent_svd_variance_to_keep=97.5):
    """Does novelty detection.

    Specifically, this method follows the procedure in Wagstaff et al. (2018)
    to determine which images in the test set are most novel with respect to the
    baseline set.

    NOTE: Both input and output images are (assumed to be) denormalized.

    B = number of baseline examples (storm objects)
    T = number of test examples (storm objects)
    M = number of rows in each storm-centered grid
    N = number of columns in each storm-centered grid
    C = number of channels (predictor variables)

    :param baseline_image_matrix: B-by-M-by-N-by-C numpy array of baseline
        images.
    :param test_image_matrix: T-by-M-by-N-by-C numpy array of test images.
    :param image_normalization_dict: See doc for `normalize_images`.
    :param predictor_names: length-C list of predictor names.
    :param cnn_model_object: Trained CNN model (instance of
        `keras.models.Model`).  Will be used to turn images into scalar
        features.
    :param cnn_feature_layer_name: The "scalar features" will be the set of
        activations from this layer.
    :param ucn_model_object: Trained UCN model (instance of
        `keras.models.Model`).  Will be used to turn scalar features into
        images.
    :param num_novel_test_images: Number of novel test images to find.
    :param percent_svd_variance_to_keep: See doc for `_fit_svd`.

    :return: novelty_dict: Dictionary with the following keys.  In the following
        discussion, Q = number of novel test images found.
    novelty_dict['novel_image_matrix_actual']: Q-by-M-by-N-by-C numpy array of
        novel test images.
    novelty_dict['novel_image_matrix_upconv']: Same as
        "novel_image_matrix_actual" but reconstructed by the upconvnet.
    novelty_dict['novel_image_matrix_upconv_svd']: Same as
        "novel_image_matrix_actual" but reconstructed by SVD (singular-value
        decomposition) and the upconvnet.

    :raises: TypeError: if `image_normalization_dict is None`.
    """

    if image_normalization_dict is None:
        error_string = (
            'image_normalization_dict cannot be None.  Must be specified.')
        raise TypeError(error_string)

    num_test_examples = test_image_matrix.shape[0]

    baseline_image_matrix_norm, _ = normalize_images(
        predictor_matrix=baseline_image_matrix + 0.,
        predictor_names=predictor_names,
        normalization_dict=image_normalization_dict)

    test_image_matrix_norm, _ = normalize_images(
        predictor_matrix=test_image_matrix + 0.,
        predictor_names=predictor_names,
        normalization_dict=image_normalization_dict)

    baseline_feature_matrix = _apply_cnn(
        cnn_model_object=cnn_model_object,
        predictor_matrix=baseline_image_matrix_norm, verbose=False,
        output_layer_name=cnn_feature_layer_name)

    test_feature_matrix = _apply_cnn(
        cnn_model_object=cnn_model_object,
        predictor_matrix=test_image_matrix_norm, verbose=False,
        output_layer_name=cnn_feature_layer_name)

    novel_indices = []
    novel_image_matrix_upconv = None
    novel_image_matrix_upconv_svd = None

    for k in range(num_novel_test_images):
        print('Finding {0:d}th of {1:d} novel test images...'.format(
            k + 1, num_novel_test_images))

        if len(novel_indices) == 0:
            this_baseline_feature_matrix = baseline_feature_matrix + 0.
            this_test_feature_matrix = test_feature_matrix + 0.
        else:
            novel_indices_numpy = numpy.array(novel_indices, dtype=int)
            this_baseline_feature_matrix = numpy.concatenate(
                (baseline_feature_matrix,
                 test_feature_matrix[novel_indices_numpy, ...]),
                axis=0)

            this_test_feature_matrix = numpy.delete(
                test_feature_matrix, obj=novel_indices_numpy, axis=0)

        svd_dictionary = _fit_svd(
            baseline_feature_matrix=this_baseline_feature_matrix,
            test_feature_matrix=this_test_feature_matrix,
            percent_variance_to_keep=percent_svd_variance_to_keep)

        svd_errors = numpy.full(num_test_examples, numpy.nan)
        test_feature_matrix_svd = numpy.full(
            test_feature_matrix.shape, numpy.nan)

        for i in range(num_test_examples):
            print(i)
            if i in novel_indices:
                continue

            test_feature_matrix_svd[i, ...] = _apply_svd(
                feature_vector=test_feature_matrix[i, ...],
                svd_dictionary=svd_dictionary)

            svd_errors[i] = numpy.linalg.norm(
                test_feature_matrix_svd[i, ...] - test_feature_matrix[i, ...]
            )

        new_novel_index = numpy.nanargmax(svd_errors)
        novel_indices.append(new_novel_index)

        new_image_matrix_upconv = ucn_model_object.predict(
            test_feature_matrix[[new_novel_index], ...], batch_size=1)

        new_image_matrix_upconv_svd = ucn_model_object.predict(
            test_feature_matrix_svd[[new_novel_index], ...], batch_size=1)

        if novel_image_matrix_upconv is None:
            novel_image_matrix_upconv = new_image_matrix_upconv + 0.
            novel_image_matrix_upconv_svd = new_image_matrix_upconv_svd + 0.
        else:
            novel_image_matrix_upconv = numpy.concatenate(
                (novel_image_matrix_upconv, new_image_matrix_upconv), axis=0)
            novel_image_matrix_upconv_svd = numpy.concatenate(
                (novel_image_matrix_upconv_svd, new_image_matrix_upconv_svd),
                axis=0)

    novel_indices = numpy.array(novel_indices, dtype=int)

    novel_image_matrix_upconv = denormalize_images(
        predictor_matrix=novel_image_matrix_upconv,
        predictor_names=predictor_names,
        normalization_dict=image_normalization_dict)

    novel_image_matrix_upconv_svd = denormalize_images(
        predictor_matrix=novel_image_matrix_upconv_svd,
        predictor_names=predictor_names,
        normalization_dict=image_normalization_dict)

    return {
        NOVEL_IMAGES_ACTUAL_KEY: test_image_matrix[novel_indices, ...],
        NOVEL_IMAGES_UPCONV_KEY: novel_image_matrix_upconv,
        NOVEL_IMAGES_UPCONV_SVD_KEY: novel_image_matrix_upconv_svd
    }


def _plot_novelty_for_many_predictors(
        novelty_matrix, predictor_names, max_absolute_temp_kelvins,
        max_absolute_refl_dbz):
    """Plots novelty for each predictor on 2-D grid with wind barbs overlain.

    M = number of rows in grid
    N = number of columns in grid
    C = number of predictors

    :param novelty_matrix: M-by-N-by-C numpy array of novelty values.
    :param predictor_names: length-C list of predictor names.
    :param max_absolute_temp_kelvins: Max absolute temperature in colour scheme.
        Minimum temperature in colour scheme will be
        -1 * `max_absolute_temp_kelvins`, and this will be a diverging scheme
        centered at zero.
    :param max_absolute_refl_dbz: Same but for reflectivity.
    :return: figure_object: See doc for `_init_figure_panels`.
    :return: axes_objects_2d_list: Same.
    """

    u_wind_matrix_m_s01 = novelty_matrix[
        ..., predictor_names.index(U_WIND_NAME)]
    v_wind_matrix_m_s01 = novelty_matrix[
        ..., predictor_names.index(V_WIND_NAME)]

    non_wind_predictor_names = [
        p for p in predictor_names if p not in [U_WIND_NAME, V_WIND_NAME]
    ]

    figure_object, axes_objects_2d_list = _init_figure_panels(
        num_rows=len(non_wind_predictor_names), num_columns=1)

    for m in range(len(non_wind_predictor_names)):
        this_predictor_index = predictor_names.index(
            non_wind_predictor_names[m])

        if non_wind_predictor_names[m] == REFLECTIVITY_NAME:
            this_min_colour_value = -1 * max_absolute_refl_dbz
            this_max_colour_value = max_absolute_refl_dbz + 0.
            this_colour_map_object = pyplot.cm.PuOr
        else:
            this_min_colour_value = -1 * max_absolute_temp_kelvins
            this_max_colour_value = max_absolute_temp_kelvins + 0.
            this_colour_map_object = pyplot.cm.bwr

        this_colour_bar_object = plot_predictor_2d(
            predictor_matrix=novelty_matrix[..., this_predictor_index],
            colour_map_object=this_colour_map_object, colour_norm_object=None,
            min_colour_value=this_min_colour_value,
            max_colour_value=this_max_colour_value,
            axes_object=axes_objects_2d_list[m][0])

        plot_wind_2d(u_wind_matrix_m_s01=u_wind_matrix_m_s01,
                     v_wind_matrix_m_s01=v_wind_matrix_m_s01,
                     axes_object=axes_objects_2d_list[m][0])

        this_colour_bar_object.set_label(non_wind_predictor_names[m])

    return figure_object, axes_objects_2d_list


def plot_novelty_detection(image_dict, novelty_dict, test_index):
    """Plots results of novelty detection.

    :param image_dict: Dictionary created by `read_many_image_files`, containing
        input data for novelty detection.
    :param novelty_dict: Dictionary created by `do_novelty_detection`,
        containing results.
    :param test_index: Array index.  The [i]th-most novel test example will be
        plotted, where i = `test_index`.
    """

    predictor_names = image_dict[PREDICTOR_NAMES_KEY]
    temperature_index = predictor_names.index(TEMPERATURE_NAME)
    reflectivity_index = predictor_names.index(REFLECTIVITY_NAME)

    image_matrix_actual = novelty_dict[NOVEL_IMAGES_ACTUAL_KEY][test_index, ...]
    image_matrix_upconv = novelty_dict[NOVEL_IMAGES_UPCONV_KEY][test_index, ...]
    image_matrix_upconv_svd = novelty_dict[
        NOVEL_IMAGES_UPCONV_SVD_KEY][test_index, ...]

    combined_matrix_kelvins = numpy.concatenate(
        (image_matrix_actual[..., temperature_index],
         image_matrix_upconv[..., temperature_index]),
        axis=0)

    min_colour_temp_kelvins = numpy.percentile(combined_matrix_kelvins, 1)
    max_colour_temp_kelvins = numpy.percentile(combined_matrix_kelvins, 99)

    this_figure_object, _ = plot_many_predictors_with_barbs(
        predictor_matrix=image_matrix_actual, predictor_names=predictor_names,
        min_colour_temp_kelvins=min_colour_temp_kelvins,
        max_colour_temp_kelvins=max_colour_temp_kelvins)

    base_title_string = '{0:d}th-most novel example'.format(test_index + 1)
    this_title_string = '{0:s}: actual'.format(base_title_string)
    this_figure_object.suptitle(this_title_string)
    pyplot.show()

    this_figure_object, _ = plot_many_predictors_with_barbs(
        predictor_matrix=image_matrix_upconv,
        predictor_names=predictor_names,
        min_colour_temp_kelvins=min_colour_temp_kelvins,
        max_colour_temp_kelvins=max_colour_temp_kelvins)

    this_title_string = r'{0:s}: upconvnet reconstruction'.format(
        base_title_string)
    this_title_string += r' ($\mathbf{X}_{up}$)'
    this_figure_object.suptitle(this_title_string)
    pyplot.show()

    novelty_matrix = image_matrix_upconv - image_matrix_upconv_svd
    max_absolute_temp_kelvins = numpy.percentile(
        numpy.absolute(novelty_matrix[..., temperature_index]), 99)
    max_absolute_refl_dbz = numpy.percentile(
        numpy.absolute(novelty_matrix[..., reflectivity_index]), 99)

    this_figure_object, _ = _plot_novelty_for_many_predictors(
        novelty_matrix=novelty_matrix, predictor_names=predictor_names,
        max_absolute_temp_kelvins=max_absolute_temp_kelvins,
        max_absolute_refl_dbz=max_absolute_refl_dbz)

    this_title_string = r'{0:s}: novelty'.format(
        base_title_string)
    this_title_string += r' ($\mathbf{X}_{up} - \mathbf{X}_{up,svd}$)'
    this_figure_object.suptitle(this_title_string)
    pyplot.show()


def do_novelty_detection_example(
        validation_image_dict, normalization_dict, cnn_model_object,
        ucn_model_object):
    """Runs novelty detection.

    The baseline images are a random set of 100 from the validation set, and the
    test images are the 100 storm objects with greatest vorticity in the
    validation set.

    :param validation_image_dict: Dictionary created by `read_many_image_files`.
    :param normalization_dict: Dictionary created by
        `get_image_normalization_params`.
    :param cnn_model_object: Trained instance of `keras.models.Model`,
        representing the CNN or "encoder".
    :param ucn_model_object: Trained instance of `keras.models.Model`,
        representing the UCN or "decoder".
    """

    target_matrix_s01 = validation_image_dict[TARGET_MATRIX_KEY]
    num_examples = target_matrix_s01.shape[0]

    max_target_by_example_s01 = numpy.array(
        [numpy.max(target_matrix_s01[i, ...]) for i in range(num_examples)]
    )

    test_indices = numpy.argsort(-1 * max_target_by_example_s01)[:100]
    test_indices = test_indices[test_indices >= 100]
    baseline_indices = numpy.linspace(0, 100, num=100, dtype=int)

    novelty_dict = do_novelty_detection(
        baseline_image_matrix=validation_image_dict[
            PREDICTOR_MATRIX_KEY][baseline_indices, ...],
        test_image_matrix=validation_image_dict[
            PREDICTOR_MATRIX_KEY][test_indices, ...],
        image_normalization_dict=normalization_dict,
        predictor_names=validation_image_dict[PREDICTOR_NAMES_KEY],
        cnn_model_object=cnn_model_object,
        cnn_feature_layer_name=get_cnn_flatten_layer(cnn_model_object),
        ucn_model_object=ucn_model_object,
        num_novel_test_images=4)


def plot_novelty_detection_example1(validation_image_dict, novelty_dict):
    """Plots first-most novel example, selon novelty detection.

    :param validation_image_dict: Dictionary created by `read_many_image_files`.
    :param novelty_dict: Dictionary created by `do_novelty_detection`.
    """

    plot_novelty_detection(image_dict=validation_image_dict,
                           novelty_dict=novelty_dict, test_index=0)


def plot_novelty_detection_example2(validation_image_dict, novelty_dict):
    """Plots second-most novel example, selon novelty detection.

    :param validation_image_dict: Dictionary created by `read_many_image_files`.
    :param novelty_dict: Dictionary created by `do_novelty_detection`.
    """

    plot_novelty_detection(image_dict=validation_image_dict,
                           novelty_dict=novelty_dict, test_index=1)


def plot_novelty_detection_example3(validation_image_dict, novelty_dict):
    """Plots third-most novel example, selon novelty detection.

    :param validation_image_dict: Dictionary created by `read_many_image_files`.
    :param novelty_dict: Dictionary created by `do_novelty_detection`.
    """

    plot_novelty_detection(image_dict=validation_image_dict,
                           novelty_dict=novelty_dict, test_index=2)


def plot_novelty_detection_example4(validation_image_dict, novelty_dict):
    """Plots fourth-most novel example, selon novelty detection.

    :param validation_image_dict: Dictionary created by `read_many_image_files`.
    :param novelty_dict: Dictionary created by `do_novelty_detection`.
    """

    plot_novelty_detection(image_dict=validation_image_dict,
                           novelty_dict=novelty_dict, test_index=3)
