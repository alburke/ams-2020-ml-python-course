"""Helper methods for normalization of predictors."""

import numpy
from interpretation import utils

NUM_VALUES_KEY = 'num_values'
MEAN_VALUE_KEY = 'mean_value'
MEAN_OF_SQUARES_KEY = 'mean_of_squares'


def _update_normalization_params(intermediate_normalization_dict, new_values):
    """Updates normalization params for one predictor.

    :param intermediate_normalization_dict: Dictionary with the following keys.
    intermediate_normalization_dict['num_values']: Number of values on which
        current estimates are based.
    intermediate_normalization_dict['mean_value']: Current estimate for mean.
    intermediate_normalization_dict['mean_of_squares']: Current mean of squared
        values.

    :param new_values: numpy array of new values (will be used to update
        `intermediate_normalization_dict`).
    :return: intermediate_normalization_dict: Same as input but with updated
        values.
    """

    if MEAN_VALUE_KEY not in intermediate_normalization_dict:
        intermediate_normalization_dict = {
            NUM_VALUES_KEY: 0,
            MEAN_VALUE_KEY: 0.,
            MEAN_OF_SQUARES_KEY: 0.
        }

    # Update mean value.
    these_means = numpy.array([
        intermediate_normalization_dict[MEAN_VALUE_KEY], numpy.mean(new_values)
    ])
    these_weights = numpy.array([
        intermediate_normalization_dict[NUM_VALUES_KEY], new_values.size
    ])
    intermediate_normalization_dict[MEAN_VALUE_KEY] = numpy.average(
        these_means, weights=these_weights
    )

    # Update mean of squares.
    these_means = numpy.array([
        intermediate_normalization_dict[MEAN_OF_SQUARES_KEY],
        numpy.mean(new_values ** 2)
    ])
    intermediate_normalization_dict[MEAN_OF_SQUARES_KEY] = numpy.average(
        these_means, weights=these_weights
    )

    # Update number of values.
    intermediate_normalization_dict[NUM_VALUES_KEY] += new_values.size

    return intermediate_normalization_dict


def _get_standard_deviation(intermediate_normalization_dict):
    """Computes stdev from intermediate normalization params.

    :param intermediate_normalization_dict: See doc for
        `_update_normalization_params`.
    :return: standard_deviation: Standard deviation.
    """

    num_values = float(intermediate_normalization_dict[NUM_VALUES_KEY])
    multiplier = num_values / (num_values - 1)

    return numpy.sqrt(multiplier * (
        intermediate_normalization_dict[MEAN_OF_SQUARES_KEY] -
        intermediate_normalization_dict[MEAN_VALUE_KEY] ** 2
    ))


def get_image_normalization_params(image_file_names):
    """Computes normalization params (mean and stdev) for each predictor.

    :param image_file_names: 1-D list of paths to input files.
    :return: normalization_dict: See input doc for `normalize_images`.
    """

    predictor_names = None
    norm_dict_by_predictor = None

    for this_file_name in image_file_names:
        print('Reading data from: "{0:s}"...'.format(this_file_name))
        this_image_dict = utils.read_image_file(this_file_name)

        if predictor_names is None:
            predictor_names = this_image_dict[utils.PREDICTOR_NAMES_KEY]
            norm_dict_by_predictor = [{}] * len(predictor_names)

        for k in range(len(predictor_names)):
            norm_dict_by_predictor[k] = _update_normalization_params(
                intermediate_normalization_dict=norm_dict_by_predictor[k],
                new_values=this_image_dict[utils.PREDICTOR_MATRIX_KEY][..., k]
            )

    print('\n')
    normalization_dict = {}

    for k in range(len(predictor_names)):
        this_mean = norm_dict_by_predictor[k][MEAN_VALUE_KEY]
        this_stdev = _get_standard_deviation(norm_dict_by_predictor[k])

        normalization_dict[predictor_names[k]] = numpy.array([
            this_mean, this_stdev
        ])

        print((
            'Mean and standard deviation for "{0:s}" = {1:.4f}, {2:.4f}'
        ).format(
            predictor_names[k], this_mean, this_stdev
        ))

    return normalization_dict


def normalize_images(
        predictor_matrix, predictor_names, normalization_dict=None):
    """Normalizes images to z-scores.

    E = number of examples (storm objects) in file
    M = number of rows in each storm-centered grid
    N = number of columns in each storm-centered grid
    C = number of channels (predictor variables)

    :param predictor_matrix: E-by-M-by-N-by-C numpy array of predictor values.
    :param predictor_names: length-C list of predictor names.
    :param normalization_dict: Dictionary.  Each key is the name of a predictor
        value, and the corresponding value is a length-2 numpy array with
        [mean, standard deviation].  If `normalization_dict is None`, mean and
        standard deviation will be computed for each predictor.
    :return: predictor_matrix: Normalized version of input.
    :return: normalization_dict: See doc for input variable.  If input was None,
        this will be a newly created dictionary.  Otherwise, this will be the
        same dictionary passed as input.
    """

    num_predictors = len(predictor_names)

    if normalization_dict is None:
        normalization_dict = {}

        for k in range(num_predictors):
            this_mean = numpy.mean(predictor_matrix[..., k])
            this_stdev = numpy.std(predictor_matrix[..., k], ddof=1)

            normalization_dict[predictor_names[k]] = numpy.array([
                this_mean, this_stdev
            ])

    for k in range(num_predictors):
        this_mean = normalization_dict[predictor_names[k]][0]
        this_stdev = normalization_dict[predictor_names[k]][1]

        predictor_matrix[..., k] = (
            (predictor_matrix[..., k] - this_mean) / float(this_stdev)
        )

    return predictor_matrix, normalization_dict


def denormalize_images(predictor_matrix, predictor_names, normalization_dict):
    """Denormalizes images from z-scores back to original scales.

    :param predictor_matrix: See doc for `normalize_images`.
    :param predictor_names: Same.
    :param normalization_dict: Same.
    :return: predictor_matrix: Denormalized version of input.
    """

    num_predictors = len(predictor_names)

    for k in range(num_predictors):
        this_mean = normalization_dict[predictor_names[k]][0]
        this_stdev = normalization_dict[predictor_names[k]][1]

        predictor_matrix[..., k] = (
            this_mean + this_stdev * predictor_matrix[..., k]
        )

    return predictor_matrix
