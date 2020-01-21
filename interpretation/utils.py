"""Helper methods for model interpretation in general."""

import copy
import glob
import errno
import time
import calendar
import os.path
import numpy
import netCDF4
from scipy.interpolate import interp1d
from scipy.ndimage.filters import gaussian_filter

DATE_FORMAT = '%Y%m%d'
DATE_FORMAT_REGEX = '[0-9][0-9][0-9][0-9][0-1][0-9][0-3][0-9]'

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

STORM_IDS_KEY = 'storm_ids'
STORM_STEPS_KEY = 'storm_steps'
PREDICTOR_NAMES_KEY = 'predictor_names'
PREDICTOR_MATRIX_KEY = 'predictor_matrix'
TARGET_NAME_KEY = 'target_name'
TARGET_MATRIX_KEY = 'target_matrix'

HIT_INDICES_KEY = 'hit_indices'
MISS_INDICES_KEY = 'miss_indices'
FALSE_ALARM_INDICES_KEY = 'false_alarm_indices'
CORRECT_NULL_INDICES_KEY = 'correct_null_indices'


def _image_file_name_to_date(netcdf_file_name):
    """Parses date from name of image (NetCDF) file.

    :param netcdf_file_name: Path to input file.
    :return: date_string: Date (format "yyyymmdd").
    """

    pathless_file_name = os.path.split(netcdf_file_name)[-1]

    date_string = pathless_file_name.replace(
        'NCARSTORM_', ''
    ).replace('-0000_d01_model_patches.nc', '')

    # Verify.
    time_string_to_unix(time_string=date_string, time_format=DATE_FORMAT)
    return date_string


def create_directory(directory_name=None, file_name=None):
    """Creates directory if necessary (i.e., doesn't already exist).

    This method checks for the argument `directory_name` first.  If
    `directory_name` is None, this method checks for `file_name` and extracts
    the directory.

    :param directory_name: Path to local directory.
    :param file_name: Path to local file.
    """

    if directory_name is None:
        directory_name = os.path.dirname(file_name)

    if directory_name == '':
        return

    try:
        os.makedirs(directory_name)
    except OSError as this_error:
        if this_error.errno == errno.EEXIST and os.path.isdir(directory_name):
            pass
        else:
            raise


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


def time_string_to_unix(time_string, time_format):
    """Converts time from string to Unix format.

    Unix format = seconds since 0000 UTC 1 Jan 1970.

    :param time_string: Time string.
    :param time_format: Format of time string (example: "%Y%m%d" or
        "%Y-%m-%d-%H%M%S").
    :return: unix_time_sec: Time in Unix format.
    """

    return calendar.timegm(time.strptime(time_string, time_format))


def time_unix_to_string(unix_time_sec, time_format):
    """Converts time from Unix format to string.

    Unix format = seconds since 0000 UTC 1 Jan 1970.

    :param unix_time_sec: Time in Unix format.
    :param time_format: Desired format of time string (example: "%Y%m%d" or
        "%Y-%m-%d-%H%M%S").
    :return: time_string: Time string.
    """

    return time.strftime(time_format, time.gmtime(unix_time_sec))


def find_many_image_files(first_date_string, last_date_string, image_dir_name):
    """Finds image (NetCDF) files in the given date range.

    :param first_date_string: First date ("yyyymmdd") in range.
    :param last_date_string: Last date ("yyyymmdd") in range.
    :param image_dir_name: Name of directory with image (NetCDF) files.
    :return: netcdf_file_names: 1-D list of paths to image files.
    """

    first_time_unix_sec = time_string_to_unix(
        time_string=first_date_string, time_format=DATE_FORMAT
    )
    last_time_unix_sec = time_string_to_unix(
        time_string=last_date_string, time_format=DATE_FORMAT
    )

    netcdf_file_pattern = (
        '{0:s}/NCARSTORM_{1:s}-0000_d01_model_patches.nc'
    ).format(image_dir_name, DATE_FORMAT_REGEX)

    netcdf_file_names = glob.glob(netcdf_file_pattern)
    netcdf_file_names.sort()

    file_date_strings = [_image_file_name_to_date(f) for f in netcdf_file_names]
    file_times_unix_sec = numpy.array([
        time_string_to_unix(time_string=d, time_format=DATE_FORMAT)
        for d in file_date_strings
    ], dtype=int)

    good_indices = numpy.where(numpy.logical_and(
        file_times_unix_sec >= first_time_unix_sec,
        file_times_unix_sec <= last_time_unix_sec
    ))[0]

    return [netcdf_file_names[k] for k in good_indices]


def read_image_file(netcdf_file_name):
    """Reads storm-centered images from NetCDF file.

    E = number of examples (storm objects) in file
    M = number of rows in each storm-centered grid
    N = number of columns in each storm-centered grid
    C = number of channels (predictor variables)

    :param netcdf_file_name: Path to input file.
    :return: image_dict: Dictionary with the following keys.
    image_dict['storm_ids']: length-E list of storm IDs (integers).
    image_dict['storm_steps']: length-E numpy array of storm steps (integers).
    image_dict['predictor_names']: length-C list of predictor names.
    image_dict['predictor_matrix']: E-by-M-by-N-by-C numpy array of predictor
        values.
    image_dict['target_name']: Name of target variable.
    image_dict['target_matrix']: E-by-M-by-N numpy array of target values.
    """

    dataset_object = netCDF4.Dataset(netcdf_file_name)

    storm_ids = numpy.array(
        dataset_object.variables[NETCDF_TRACK_ID_NAME][:], dtype=int
    )
    storm_steps = numpy.array(
        dataset_object.variables[NETCDF_TRACK_STEP_NAME][:], dtype=int
    )

    predictor_matrix = None

    for this_predictor_name in NETCDF_PREDICTOR_NAMES:
        this_predictor_matrix = numpy.array(
            dataset_object.variables[this_predictor_name][:], dtype=float
        )
        this_predictor_matrix = numpy.expand_dims(
            this_predictor_matrix, axis=-1
        )

        if predictor_matrix is None:
            predictor_matrix = this_predictor_matrix + 0.
        else:
            predictor_matrix = numpy.concatenate(
                (predictor_matrix, this_predictor_matrix), axis=-1
            )

    target_matrix = numpy.array(
        dataset_object.variables[NETCDF_TARGET_NAME][:], dtype=float
    )

    return {
        STORM_IDS_KEY: storm_ids,
        STORM_STEPS_KEY: storm_steps,
        PREDICTOR_NAMES_KEY: PREDICTOR_NAMES,
        PREDICTOR_MATRIX_KEY: predictor_matrix,
        TARGET_NAME_KEY: TARGET_NAME,
        TARGET_MATRIX_KEY: target_matrix
    }


def read_many_image_files(netcdf_file_names):
    """Reads storm-centered images from many NetCDF files.

    :param netcdf_file_names: 1-D list of paths to input files.
    :return: image_dict: See doc for `read_image_file`.
    """

    image_dict = None
    keys_to_concat = [
        STORM_IDS_KEY, STORM_STEPS_KEY, PREDICTOR_MATRIX_KEY, TARGET_MATRIX_KEY
    ]

    for this_file_name in netcdf_file_names:
        print('Reading data from: "{0:s}"...'.format(this_file_name))
        this_image_dict = read_image_file(this_file_name)

        if image_dict is None:
            image_dict = copy.deepcopy(this_image_dict)
            continue

        for this_key in keys_to_concat:
            image_dict[this_key] = numpy.concatenate((
                image_dict[this_key], this_image_dict[this_key]
            ), axis=0)

    return image_dict


def find_extreme_examples(
        class_labels, event_probabilities, num_examples_per_set):
    """Finds extreme examples.

    There are four sets of examples:

    - best hits
    - worst false alarms
    - worst misses
    - best correct nulls

    E = total number of examples
    e = number of examples per set

    :param class_labels: length-E numpy array of class labels (1 for event, 0
        for non-event).
    :param event_probabilities: length-E numpy array of event probabilities.
    :param num_examples_per_set: Number of examples in each set.

    :return: extreme_dict: Dictionary with the following keys.
    extreme_dict['hit_indices']: length-e numpy array with indices of best hits.
    extreme_dict['miss_indices']: length-e numpy array with indices of worst
        misses.
    extreme_dict['false_alarm_indices']: length-e numpy array with indices of
        worst false alarms.
    extreme_dict['correct_null_indices']: length-e numpy array with indices of
        best correct nulls.
    """

    # Check input args.
    class_labels = numpy.round(class_labels).astype(int)
    assert numpy.all(class_labels >= 0)
    assert numpy.all(class_labels <= 1)
    assert len(class_labels.shape) == 1

    num_examples_total = len(class_labels)

    assert numpy.all(event_probabilities >= 0.)
    assert numpy.all(event_probabilities <= 1.)
    assert len(event_probabilities.shape) == 1
    assert len(event_probabilities) == num_examples_total

    num_examples_per_set = int(numpy.round(num_examples_per_set))
    assert num_examples_per_set > 0

    positive_indices = numpy.where(class_labels == 1)[0]
    negative_indices = numpy.where(class_labels == 0)[0]

    num_hits = min([
        num_examples_per_set, len(positive_indices)
    ])
    num_misses = min([
        num_examples_per_set, len(positive_indices)
    ])
    num_false_alarms = min([
        num_examples_per_set, len(negative_indices)
    ])
    num_correct_nulls = min([
        num_examples_per_set, len(negative_indices)
    ])

    these_indices = numpy.argsort(-1 * event_probabilities[positive_indices])
    hit_indices = positive_indices[these_indices][:num_hits]
    print('Average event probability for {0:d} best hits = {1:.4f}'.format(
        num_hits, numpy.mean(event_probabilities[hit_indices])
    ))

    these_indices = numpy.argsort(event_probabilities[positive_indices])
    miss_indices = positive_indices[these_indices][:num_misses]
    print('Average event probability for {0:d} worst misses = {1:.4f}'.format(
        num_misses, numpy.mean(event_probabilities[miss_indices])
    ))

    these_indices = numpy.argsort(-1 * event_probabilities[negative_indices])
    false_alarm_indices = negative_indices[these_indices][:num_false_alarms]
    print((
        'Average event probability for {0:d} worst false alarms = {1:.4f}'
    ).format(
        num_false_alarms, numpy.mean(event_probabilities[false_alarm_indices])
    ))

    these_indices = numpy.argsort(event_probabilities[negative_indices])
    correct_null_indices = negative_indices[these_indices][:num_correct_nulls]
    print((
        'Average event probability for {0:d} best correct nulls = {1:.4f}'
    ).format(
        num_correct_nulls, numpy.mean(event_probabilities[correct_null_indices])
    ))

    return {
        HIT_INDICES_KEY: hit_indices,
        MISS_INDICES_KEY: miss_indices,
        FALSE_ALARM_INDICES_KEY: false_alarm_indices,
        CORRECT_NULL_INDICES_KEY: correct_null_indices
    }


def run_pmm_one_variable(field_matrix, max_percentile_level=99.):
    """Applies PMM (probability-matched means) to one variable.

    :param field_matrix: numpy array with data to be averaged.  The first axis
        should represent examples, and remaining axes should represent spatial
        dimensions.
    :param max_percentile_level: Maximum percentile.  No output value will
        exceed the [q]th percentile of `field_matrix`, where q =
        `max_percentile_level`.  Similarly, no output value will be less than
        the [100 - q]th percentile of `field_matrix`.
    :return: mean_field_matrix: numpy array with average spatial field.
        Dimensions are the same as `field_matrix`, except that the first axis is
        gone.  For instance, if `field_matrix` is 1000 x 32 x 32 (1000 examples
        x 32 rows x 32 columns), `mean_field_matrix` will be 32 x 32.
    """

    assert not numpy.any(numpy.isnan(field_matrix))
    assert len(field_matrix.shape) > 1
    assert max_percentile_level >= 90.
    assert max_percentile_level < 100.

    # Pool values over all dimensions and remove extremes.
    pooled_values = numpy.sort(numpy.ravel(field_matrix))
    max_pooled_value = numpy.percentile(pooled_values, max_percentile_level)
    pooled_values = pooled_values[pooled_values <= max_pooled_value]

    min_pooled_value = numpy.percentile(
        pooled_values, 100 - max_percentile_level
    )
    pooled_values = pooled_values[pooled_values >= min_pooled_value]

    # Find ensemble mean at each location (e.g., grid point).
    mean_field_matrix = numpy.mean(field_matrix, axis=0)
    mean_field_flattened = numpy.ravel(mean_field_matrix)

    # At each location, replace ensemble mean with the same percentile from the
    # pooled array.
    pooled_value_percentiles = numpy.linspace(
        0, 100, num=len(pooled_values), dtype=float
    )
    mean_value_percentiles = numpy.linspace(
        0, 100, num=len(mean_field_flattened), dtype=float
    )

    sort_indices = numpy.argsort(mean_field_flattened)
    unsort_indices = numpy.argsort(sort_indices)

    interp_object = interp1d(
        pooled_value_percentiles, pooled_values, kind='linear',
        bounds_error=True, assume_sorted=True
    )

    mean_field_flattened = interp_object(mean_value_percentiles)
    mean_field_flattened = mean_field_flattened[unsort_indices]
    mean_field_matrix = numpy.reshape(
        mean_field_flattened, mean_field_matrix.shape
    )

    return mean_field_matrix


def run_pmm_many_variables(field_matrix, max_percentile_level=99.):
    """Applies PMM (probability-matched means) to each variable.

    :param field_matrix: numpy array with data to be averaged.  The first axis
        should represent examples; the last axis should represent variables; and
        remaining axes should represent spatial dimensions.
    :param max_percentile_level: See doc for `run_pmm_one_variable`.
    :return: mean_field_matrix: numpy array with average spatial fields.
        Dimensions are the same as `field_matrix`, except that the first axis is
        gone.  For instance, if `field_matrix` is 1000 x 32 x 32 x 4
        (1000 examples x 32 rows x 32 columns x 4 variables),
        `mean_field_matrix` will be 32 x 32 x 4.
    """

    assert len(field_matrix.shape) > 2

    num_variables = field_matrix.shape[-1]
    mean_field_matrix = numpy.full(field_matrix.shape[1:], numpy.nan)

    for k in range(num_variables):
        mean_field_matrix[..., k] = run_pmm_one_variable(
            field_matrix=field_matrix[..., k],
            max_percentile_level=max_percentile_level
        )

    return mean_field_matrix
