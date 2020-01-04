"""Helper methods for novelty detection."""

import numpy
from matplotlib import pyplot
from interpretation import cnn, utils, plotting

EOF_MATRIX_KEY = 'eof_matrix'
FEATURE_MEANS_KEY = 'feature_means'
FEATURE_STDEVS_KEY = 'feature_standard_deviations'

NOVEL_INDICES_KEY = 'novel_indices'
NOVEL_MATRIX_UPCONV_KEY = 'novel_matrix_upconv'
NOVEL_MATRIX_UPCONV_SVD_KEY = 'novel_matrix_upconv_svd'

REFL_COLOUR_MAP_OBJECT = pyplot.get_cmap('PuOr')
TEMPERATURE_COLOUR_MAP_OBJECT = pyplot.get_cmap('bwr')


def _normalize_features(
        feature_matrix, feature_means=None, feature_standard_deviations=None):
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

    assert percent_variance_to_keep >= 50.
    assert percent_variance_to_keep <= 100.

    combined_feature_matrix = numpy.concatenate(
        (baseline_feature_matrix, test_feature_matrix), axis=0
    )
    combined_feature_matrix, feature_means, feature_standard_deviations = (
        _normalize_features(feature_matrix=combined_feature_matrix)
    )

    num_features = baseline_feature_matrix.shape[1]
    num_baseline_examples = baseline_feature_matrix.shape[0]
    baseline_feature_matrix = (
        combined_feature_matrix[:num_baseline_examples, ...]
    )

    eigenvalues, eof_matrix = numpy.linalg.svd(baseline_feature_matrix)[1:]
    eigenvalues = eigenvalues ** 2

    explained_variances = eigenvalues / numpy.sum(eigenvalues)
    cumulative_explained_variances = numpy.cumsum(explained_variances)

    fraction_of_variance_to_keep = 0.01 * percent_variance_to_keep
    these_indices = numpy.where(
        cumulative_explained_variances >= fraction_of_variance_to_keep
    )[0]

    if len(these_indices) == 0:
        these_indices = numpy.array([num_features - 1], dtype=int)

    num_modes_to_keep = 1 + these_indices[0]

    print((
        'Number of modes required to explain {0:f}% of variance: {1:d}'
    ).format(
        percent_variance_to_keep, num_modes_to_keep
    ))

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
        this_matrix, feature_vector_norm
    )

    return (
        svd_dictionary[FEATURE_MEANS_KEY] +
        reconstructed_feature_vector_norm * svd_dictionary[FEATURE_STDEVS_KEY]
    )


def _plot_novelty_maps(novelty_matrix, predictor_names, max_temp_diff_kelvins,
                       max_reflectivity_diff_dbz):
    """Plots novelty maps for one example.

    M = number of rows in grid
    N = number of columns in grid
    C = number of predictors

    :param novelty_matrix: M-by-N-by-C numpy array of denormalized novelty
        values (upconvnet reconstruction minus upconvnet/SVD reconstruction).
    :param predictor_names: length-C list of predictor names.
    :param max_temp_diff_kelvins: Max temperature difference in colour bar.
    :param max_reflectivity_diff_dbz: Max reflectivity difference in colour bar.
    :return: figure_object: Figure handle (instance of
        `matplotlib.figure.Figure`).
    :return: axes_object_matrix: 2-D numpy array of axes handles (instances
        of `matplotlib.axes._subplots.AxesSubplot`).
    """

    u_diff_matrix_m_s01 = novelty_matrix[
        ..., predictor_names.index(utils.U_WIND_NAME)
    ]
    v_diff_matrix_m_s01 = novelty_matrix[
        ..., predictor_names.index(utils.V_WIND_NAME)
    ]

    non_wind_predictor_names = [
        p for p in predictor_names
        if p not in [utils.U_WIND_NAME, utils.V_WIND_NAME]
    ]

    figure_object, axes_object_matrix = plotting._create_paneled_figure(
        num_rows=1, num_columns=len(non_wind_predictor_names),
    )

    for k in range(len(non_wind_predictor_names)):
        this_predictor_index = predictor_names.index(
            non_wind_predictor_names[k]
        )

        if non_wind_predictor_names[k] == utils.REFLECTIVITY_NAME:
            this_max_colour_value = max_reflectivity_diff_dbz
            this_colour_map_object = REFL_COLOUR_MAP_OBJECT
        else:
            this_max_colour_value = max_temp_diff_kelvins
            this_colour_map_object = TEMPERATURE_COLOUR_MAP_OBJECT

        plotting.plot_scalar_field_2d(
            predictor_matrix=novelty_matrix[..., this_predictor_index],
            colour_map_object=this_colour_map_object,
            min_colour_value=-this_max_colour_value,
            max_colour_value=this_max_colour_value,
            axes_object=axes_object_matrix[0, k]
        )

        this_colour_bar_object = plotting.plot_linear_colour_bar(
            axes_object_or_matrix=axes_object_matrix[0, k],
            data_values=novelty_matrix[..., this_predictor_index],
            colour_map_object=this_colour_map_object,
            min_value=-this_max_colour_value, max_value=this_max_colour_value,
            plot_horizontal=True, plot_min_arrow=True, plot_max_arrow=True
        )

        plotting.plot_wind_2d(
            u_wind_matrix_m_s01=u_diff_matrix_m_s01,
            v_wind_matrix_m_s01=v_diff_matrix_m_s01,
            axes_object=axes_object_matrix[0, k]
        )

        this_colour_bar_object.set_label(
            non_wind_predictor_names[k],
            fontsize=plotting.DEFAULT_CBAR_FONT_SIZE
        )

    return figure_object, axes_object_matrix


def run_novelty_detection(
        baseline_predictor_matrix_norm, trial_predictor_matrix_norm,
        cnn_model_object, cnn_feature_layer_name, upconvnet_model_object,
        num_novel_examples, multipass=False, percent_variance_to_keep=97.5):
    """Runs novelty detection.

    B = number of baseline examples
    T = number of trial examples
    Q = number of novel trial examples to find

    :param baseline_predictor_matrix_norm: numpy array with normalized predictor
        values for baseline set.  The first axis should have length B.
    :param trial_predictor_matrix_norm: numpy array with normalized predictor
        values for trial set.  The first axis should have length T.
    :param cnn_model_object: Trained CNN (instance of `keras.models.Model` or
        `keras.models.Sequential`).
    :param cnn_feature_layer_name: Name of feature layer in CNN.  Outputs from
        this layer will be inputs to the upconvnet.
    :param upconvnet_model_object: Trained upconvnet (instance of
        `keras.models.Model` or `keras.models.Sequential`).
    :param num_novel_examples: Q in the above discussion.
    :param multipass: Boolean flag.  If True, will run multi-pass version.  If
        False, will run single-pass version.  In the multi-pass version,
        whenever the next-most novel trial example is found, it is used to fit a
        new SVD model.  In other words, after finding the [i]th-most novel trial
        example, a new SVD model is fit on all baseline examples and the i most
        novel trial examples.
    :param percent_variance_to_keep: Percentage of variance to keep in SVD
        (singular-value decomposition) from image space to feature space.
    :return: novelty_dict: Dictionary with the following keys.
    novelty_dict['novel_indices']: length-Q numpy array with indices of novel
        examples, where novel_indices[i] is the index of the [i]th-most novel
        example.
    novelty_dict['novel_matrix_upconv']: numpy array with upconvnet
        reconstructions of the most novel examples.  The first axis has length
        Q.
    novelty_dict['novel_matrix_upconv_svd']: numpy array with upconvnet
        reconstructions of SVD reconstructions of the most novel examples.
        Same dimensions as `novel_matrix_upconv`.
    """

    multipass = bool(multipass)

    num_trial_examples = trial_predictor_matrix_norm.shape[0]
    num_novel_examples = int(numpy.round(num_novel_examples))
    num_novel_examples = min([num_novel_examples, num_trial_examples])

    assert num_novel_examples > 1

    baseline_feature_matrix = cnn.apply_cnn(
        model_object=cnn_model_object,
        predictor_matrix=baseline_predictor_matrix_norm,
        output_layer_name=cnn_feature_layer_name, verbose=True
    )
    print('\n')

    trial_feature_matrix = cnn.apply_cnn(
        model_object=cnn_model_object,
        predictor_matrix=trial_predictor_matrix_norm,
        output_layer_name=cnn_feature_layer_name, verbose=True
    )
    print('\n')

    svd_dictionary = None
    novel_indices = numpy.array([], dtype=int)
    novel_matrix_upconv = None
    novel_matrix_upconv_svd = None

    for i in range(num_novel_examples):
        print('Finding {0:d}th-most novel trial example...'.format(
            i + 1, num_novel_examples
        ))

        fit_new_svd = multipass or i == 0

        if fit_new_svd:
            this_baseline_feature_matrix = numpy.concatenate((
                baseline_feature_matrix,
                trial_feature_matrix[novel_indices, ...]
            ), axis=0)

            this_trial_feature_matrix = numpy.delete(
                trial_feature_matrix, obj=novel_indices, axis=0
            )

            svd_dictionary = _fit_svd(
                baseline_feature_matrix=this_baseline_feature_matrix,
                test_feature_matrix=this_trial_feature_matrix,
                percent_variance_to_keep=percent_variance_to_keep
            )

        trial_svd_errors = numpy.full(num_trial_examples, numpy.nan)
        trial_feature_matrix_svd = numpy.full(
            trial_feature_matrix.shape, numpy.nan
        )

        for i in range(num_trial_examples):
            if i in novel_indices:
                continue

            trial_feature_matrix_svd[i, ...] = _apply_svd(
                feature_vector=trial_feature_matrix[i, ...],
                svd_dictionary=svd_dictionary
            )

            trial_svd_errors[i] = numpy.linalg.norm(
                trial_feature_matrix_svd[i, ...] - trial_feature_matrix[i, ...]
            )

        this_novel_index = numpy.nanargmax(trial_svd_errors)
        this_novel_index_array = numpy.array([this_novel_index], dtype=int)
        novel_indices = numpy.concatenate((
            novel_indices, this_novel_index_array
        ))

        this_image_matrix_upconv = upconvnet_model_object.predict(
            trial_feature_matrix[this_novel_index_array, ...], batch_size=1
        )

        this_image_matrix_upconv_svd = upconvnet_model_object.predict(
            trial_feature_matrix_svd[this_novel_index_array, ...], batch_size=1
        )

        if novel_matrix_upconv is None:
            these_dim = (
                (num_novel_examples,) + this_image_matrix_upconv.shape[1:]
            )
            novel_matrix_upconv = numpy.full(these_dim, numpy.nan)
            novel_matrix_upconv_svd = numpy.full(these_dim, numpy.nan)

        novel_matrix_upconv[i, ...] = this_image_matrix_upconv
        novel_matrix_upconv_svd[i, ...] = this_image_matrix_upconv_svd

    return {
        NOVEL_INDICES_KEY: novel_indices,
        NOVEL_MATRIX_UPCONV_KEY: novel_matrix_upconv,
        NOVEL_MATRIX_UPCONV_SVD_KEY: novel_matrix_upconv_svd
    }


def plot_results(trial_predictor_matrix_denorm, predictor_names,
                 novelty_dict_denorm, novel_index):
    """Plots results of novelty detection.

    T = number of trial examples
    M = number of rows in grid
    N = number of columns in grid
    C = number of predictors

    :param trial_predictor_matrix_denorm: T-by-M-by-N-by-C numpy array with
        denormalized predictor values.
    :param predictor_names: length-C list of predictor names.
    :param novelty_dict_denorm: Dictionary created by `run_novelty_detection`,
        except with denormalized predictor values.
    :param novel_index: Will plot the [k]th most novel trial example, where k is
        this arg.
    """

    temperature_index = predictor_names.index(utils.TEMPERATURE_NAME)
    reflectivity_index = predictor_names.index(utils.REFLECTIVITY_NAME)

    trial_index = novelty_dict_denorm[NOVEL_INDICES_KEY][novel_index]
    actual_predictor_matrix = trial_predictor_matrix_denorm[trial_index, ...]

    predictor_matrix_upconv = (
        novelty_dict_denorm[NOVEL_MATRIX_UPCONV_KEY][novel_index, ...]
    )
    predictor_matrix_upconv_svd = (
        novelty_dict_denorm[NOVEL_MATRIX_UPCONV_SVD_KEY][novel_index, ...]
    )
    novelty_matrix = predictor_matrix_upconv - predictor_matrix_upconv_svd

    concat_temp_matrix_kelvins = numpy.concatenate((
        actual_predictor_matrix[..., temperature_index],
        predictor_matrix_upconv[..., temperature_index]
    ), axis=0)

    min_colour_temp_kelvins = numpy.percentile(concat_temp_matrix_kelvins, 1)
    max_colour_temp_kelvins = numpy.percentile(concat_temp_matrix_kelvins, 99)

    figure_object, _ = plotting.plot_many_predictors_with_barbs(
        predictor_matrix=actual_predictor_matrix,
        predictor_names=predictor_names,
        min_colour_temp_kelvins=min_colour_temp_kelvins,
        max_colour_temp_kelvins=max_colour_temp_kelvins
    )

    figure_object.suptitle('Actual example')

    figure_object, _ = plotting.plot_many_predictors_with_barbs(
        predictor_matrix=predictor_matrix_upconv,
        predictor_names=predictor_names,
        min_colour_temp_kelvins=min_colour_temp_kelvins,
        max_colour_temp_kelvins=max_colour_temp_kelvins
    )

    figure_object.suptitle('Upconvnet reconstruction')

    max_temp_diff_kelvins = numpy.percentile(
        numpy.absolute(novelty_matrix[..., temperature_index]), 99
    )
    max_reflectivity_diff_dbz = numpy.percentile(
        numpy.absolute(novelty_matrix[..., reflectivity_index]), 99
    )

    figure_object, _ = _plot_novelty_maps(
        novelty_matrix=novelty_matrix, predictor_names=predictor_names,
        max_temp_diff_kelvins=max_temp_diff_kelvins,
        max_reflectivity_diff_dbz=max_reflectivity_diff_dbz
    )

    title_string = (
        'Novelty (full upconvnet reconstruction minus upconvnet\n'
        'reconstruction from baseline examples'' feature space)'
    )
    figure_object.suptitle(title_string, fontsize=12)
