"""Helper methods for the permutation importance test."""

import numpy
from matplotlib import pyplot
from sklearn.metrics import roc_auc_score as sklearn_auc

SEPARATOR_STRING = '\n\n' + '*' * 50 + '\n\n'

DEFAULT_NUM_BOOTSTRAP_REPS = 1000
DEFAULT_CONFIDENCE_LEVEL = 0.95

DEFAULT_BAR_COLOUR = numpy.array([252, 141, 98], dtype=float) / 255
NO_PERMUTATION_COLOUR = numpy.full(3, 1.)
BAR_EDGE_COLOUR = numpy.full(3, 0.)
ERROR_BAR_COLOUR = numpy.full(3, 0.)
REFERENCE_LINE_COLOUR = numpy.full(3, 152. / 255)

BAR_EDGE_WIDTH = 2
REFERENCE_LINE_WIDTH = 4
ERROR_BAR_CAP_SIZE = 8
ERROR_BAR_DICT = {'alpha': 0.5, 'linewidth': 4, 'capthick': 4}

BAR_TEXT_COLOUR = numpy.full(3, 0.)
BAR_FONT_SIZE = 17.5
DEFAULT_FONT_SIZE = 22.5
FIGURE_WIDTH_INCHES = 10
FIGURE_HEIGHT_INCHES = 10

pyplot.rc('font', size=DEFAULT_FONT_SIZE)
pyplot.rc('axes', titlesize=DEFAULT_FONT_SIZE)
pyplot.rc('axes', labelsize=DEFAULT_FONT_SIZE)
pyplot.rc('xtick', labelsize=DEFAULT_FONT_SIZE)
pyplot.rc('ytick', labelsize=DEFAULT_FONT_SIZE)
pyplot.rc('legend', fontsize=DEFAULT_FONT_SIZE)
pyplot.rc('figure', titlesize=DEFAULT_FONT_SIZE)

PREDICTOR_MATRIX_KEY = 'predictor_matrix'
PERMUTED_FLAGS_KEY = 'permuted_channel_flags'
PERMUTED_PREDICTORS_KEY = 'permuted_predictor_names'
PERMUTED_COST_MATRIX_KEY = 'permuted_cost_matrix'
UNPERMUTED_PREDICTORS_KEY = 'unpermuted_predictor_names'
UNPERMUTED_COST_MATRIX_KEY = 'unpermuted_cost_matrix'
BEST_PREDICTOR_KEY = 'best_predictor_name'
BEST_COST_ARRAY_KEY = 'best_cost_array'

BEST_PREDICTORS_KEY = 'best_predictor_names'
BEST_COST_MATRIX_KEY = 'best_cost_matrix'
ORIGINAL_COST_ARRAY_KEY = 'original_cost_array'
STEP1_PREDICTORS_KEY = 'step1_predictor_names'
STEP1_COST_MATRIX_KEY = 'step1_cost_matrix'
BACKWARDS_FLAG = 'backwards_test'


def _permute_one_channel(predictor_matrix, channel_index, permuted_values=None):
    """Permutes values in one channel.

    :param predictor_matrix: numpy array of predictor values (input to CNN).
    :param channel_index: Will permute values in the [k]th channel, where
        k = `channel_index`.
    :param permuted_values: numpy array of permuted values with which to replace
        clean values.  This should have the same dimensions as
        `predictor_matrix` but without the last (channel) dimensions.  For
        example, if `predictor_matrix` is 1000 x 32 x 32 x 4
        (1000 examples x 32 rows x 32 columns x 4 channels), `permuted_values`
        should be 1000 x 32 x 32.
    :return: predictor_matrix: Same as input but with different values.
    :return: permuted_values: numpy array of permuted values used to replace
        clean values.  See input doc for exact format of this array.
    """

    num_examples = predictor_matrix.shape[0]
    k = channel_index

    if permuted_values is None:
        predictor_matrix[..., k] = numpy.take(
            predictor_matrix[..., k],
            indices=numpy.random.permutation(num_examples),
            axis=0
        )
    else:
        predictor_matrix[..., k] = permuted_values

    permuted_values = predictor_matrix[..., k]

    return predictor_matrix, permuted_values


def _unpermute_one_channel(predictor_matrix, clean_predictor_matrix,
                           channel_index):
    """Unpermutes values in one channel.

    "Unpermuting" means restoring to the correct order.

    :param predictor_matrix: numpy array of predictor values (input to CNN).
    :param clean_predictor_matrix: Clean version of `predictor_matrix` (with no
        values permuted).
    :param channel_index: Will unpermute values in the [k]th channel, where
        k = `channel_index`.
    :return: predictor_matrix: Same as input but with clean values in the [k]th
        channel, where k = `channel_index`.
    """

    k = channel_index
    predictor_matrix[..., k] = clean_predictor_matrix[..., k]

    return predictor_matrix


def _bootstrap_cost(target_values, class_probability_matrix, cost_function,
                    num_replicates):
    """Bootstraps cost for one set of examples.

    E = number of examples
    K = number of classes
    B = number of bootstrap replicates

    :param target_values: length-E numpy array of target values (integers in
        range 0...[K - 1]).
    :param class_probability_matrix: E-by-K numpy array of predicted
        probabilities.
    :param cost_function: Cost function, used to evaluate predicted
        probabilities.  Must be negatively oriented (so that lower values are
        better), with the following inputs and outputs.
    Input: target_values: Same as input to this method.
    Input: class_probability_matrix: Same as input to this method.
    Output: cost: Scalar value.

    :param num_replicates: Number of bootstrap replicates.
    :return: cost_values: length-B numpy array of cost values.
    """

    cost_values = numpy.full(num_replicates, numpy.nan)

    if num_replicates == 1:
        cost_values[0] = cost_function(target_values,
                                       class_probability_matrix)
    else:
        num_examples = len(target_values)
        example_indices = numpy.linspace(
            0, num_examples - 1, num=num_examples, dtype=int
        )

        for k in range(num_replicates):
            these_indices = numpy.random.choice(
                example_indices, size=num_examples, replace=True
            )

            cost_values[k] = cost_function(
                target_values[these_indices],
                class_probability_matrix[these_indices, ...]
            )

    print('Average cost = {0:.4f}'.format(numpy.mean(cost_values)))
    return cost_values


def _event_probs_to_multiclass(event_probabilities):
    """Converts 1-D array of event probabilities to 2-D array.

    E = number of examples

    :param event_probabilities: length-E numpy array of event probabilities.
    :return: class_probability_matrix: E-by-2 numpy array, where second column
        contains probabilities of event and first column contains probabilities
        of non-event.
    """

    these_probs = numpy.reshape(
        event_probabilities, (len(event_probabilities), 1)
    )
    return numpy.hstack((1. - these_probs, these_probs))


def _run_forward_test_one_step(
        model_object, predictor_matrix, predictor_names, target_values,
        cost_function, num_bootstrap_reps, step_num, permuted_channel_flags):
    """Runs one step of the forward permutation test.

    E = number of examples
    C = number of channels (predictor variables)
    K = number of classes
    P = number of permutations attempted in this step
    B = number of bootstrap replicates

    :param model_object: Trained model (instance of `keras.models.Model` or
        `keras.models.Sequential`).
    :param predictor_matrix: numpy array with predictors for many examples.  The
        first axis should have length E, and the last should have length C.
    :param predictor_names: length-C list of predictor names (strings).
    :param target_values: length-E numpy array of target values.  Should all be
        integers in range 0...(K - 1).

    :param cost_function: Cost function, used to evaluate predicted
        probabilities.  Must be negatively oriented (so that lower values are
        better), with the following inputs and outputs.
    Input: target_values: Same as input to this method.
    Input: class_probability_matrix: E-by-K numpy array of predicted
        probabilities.
    Output: cost: Scalar value.

    :param num_bootstrap_reps: Number of bootstrap replicates.  Examples will be
        bootstrapped this many times to get a confidence interval for the cost.
    :param step_num: Current step (integer) in overall permutation test.
    :param permuted_channel_flags: length-C numpy array of Boolean flags,
        indicating which channels are already permuted in `predictor_matrix`.

    :return: forward_step_dict: Dictionary with the following keys.
    forward_step_dict["predictor_matrix"]: Same as input but maybe with
        different values.
    forward_step_dict["permuted_channel_flags"]: Same as input but with one more
        True flag.
    forward_step_dict["permuted_predictor_names"]: length-P list with names of
        predictors temporarily permuted in this step.
    forward_step_dict["permuted_cost_matrix"]: P-by-B numpy array of costs after
        permutation.
    forward_step_dict["best_predictor_name"]: Name of best predictor in this
        step.
    forward_step_dict["best_cost_array"]: length-B numpy array of costs after
        permutation of best predictor.
    """

    best_predictor_index = -1
    best_permuted_values = None
    best_cost_array = numpy.full(num_bootstrap_reps, -numpy.inf)

    permuted_predictor_names = []
    permuted_cost_matrix = numpy.full((0, num_bootstrap_reps), numpy.nan)

    num_channels = predictor_matrix.shape[-1]

    for k in range(num_channels):
        if permuted_channel_flags[k]:
            continue

        print('Permuting predictor "{0:s}" at step {1:d}...'.format(
            predictor_names[k], step_num
        ))

        this_predictor_matrix, these_permuted_values = _permute_one_channel(
            predictor_matrix=predictor_matrix + 0., channel_index=k
        )

        these_probs = model_object.predict(
            this_predictor_matrix, batch_size=this_predictor_matrix.shape[0]
        )
        this_probability_matrix = _event_probs_to_multiclass(these_probs)

        this_cost_array = _bootstrap_cost(
            target_values=target_values,
            class_probability_matrix=this_probability_matrix,
            cost_function=cost_function, num_replicates=num_bootstrap_reps
        )
        this_cost_matrix = numpy.reshape(
            this_cost_array, (1, len(this_cost_array))
        )

        permuted_predictor_names.append(predictor_names[k])
        permuted_cost_matrix = numpy.concatenate(
            (permuted_cost_matrix, this_cost_matrix), axis=0
        )

        if numpy.mean(this_cost_array) < numpy.mean(best_cost_array):
            continue

        best_predictor_index = k + 0
        best_permuted_values = these_permuted_values + 0.
        best_cost_array = this_cost_array + 0.

    if len(permuted_predictor_names) == 0:
        return None

    k = best_predictor_index + 0
    best_predictor_name = predictor_names[k]
    permuted_channel_flags[k] = True

    predictor_matrix = _permute_one_channel(
        predictor_matrix=predictor_matrix, channel_index=k,
        permuted_values=best_permuted_values
    )[0]

    print('Best predictor = "{0:s}" ... cost = {1:.4f}'.format(
        best_predictor_name, numpy.mean(best_cost_array)
    ))

    return {
        PREDICTOR_MATRIX_KEY: predictor_matrix,
        PERMUTED_FLAGS_KEY: permuted_channel_flags,
        PERMUTED_PREDICTORS_KEY: permuted_predictor_names,
        PERMUTED_COST_MATRIX_KEY: permuted_cost_matrix,
        BEST_PREDICTOR_KEY: best_predictor_name,
        BEST_COST_ARRAY_KEY: best_cost_array
    }


def _run_backwards_test_one_step(
        model_object, predictor_matrix, clean_predictor_matrix,
        predictor_names, target_values, cost_function,
        num_bootstrap_reps, step_num, permuted_channel_flags):
    """Runs one step of the backwards permutation test.

    P = number of unpermutations (depermutations?) attempted in this step
    B = number of bootstrap replicates

    :param model_object: See doc for `_run_forward_test_one_step`.
    :param predictor_matrix: Same.
    :param clean_predictor_matrix: Clean version of `predictor_matrix` (with no
        values permuted).
    :param predictor_names: See doc for `_run_forward_test_one_step`.
    :param target_values: Same.
    :param cost_function: Same.
    :param num_bootstrap_reps: Same.
    :param step_num: Same.
    :param permuted_channel_flags: Same.

    :return: backwards_step_dict: Dictionary with the following keys.
    backwards_step_dict["predictor_matrix"]: Same as input but maybe with
        different values.
    backwards_step_dict["permuted_channel_flags"]: Same as input but with one
        more False flag.
    backwards_step_dict["unpermuted_predictor_names"]: length-P list with names
        of predictors temporarily unpermuted in this step.
    backwards_step_dict["unpermuted_cost_matrix"]: P-by-B numpy array of costs
        after permutation.
    backwards_step_dict["best_predictor_name"]: Name of best predictor in this
        step.
    backwards_step_dict["best_cost_array"]: length-B numpy array of costs after
        unpermutation of best predictor.
    """

    best_predictor_index = -1
    best_cost_array = numpy.full(num_bootstrap_reps, numpy.inf)

    unpermuted_predictor_names = []
    unpermuted_cost_matrix = numpy.full((0, num_bootstrap_reps), numpy.nan)

    num_channels = predictor_matrix.shape[-1]

    for k in range(num_channels):
        if not permuted_channel_flags[k]:
            continue

        print('Unpermuting predictor "{0:s}" at step {1:d}...'.format(
            predictor_names[k], step_num
        ))

        this_predictor_matrix = _unpermute_one_channel(
            predictor_matrix=predictor_matrix + 0.,
            clean_predictor_matrix=clean_predictor_matrix, channel_index=k
        )

        these_probs = model_object.predict(
            this_predictor_matrix, batch_size=this_predictor_matrix.shape[0]
        )
        this_probability_matrix = _event_probs_to_multiclass(these_probs)

        this_cost_array = _bootstrap_cost(
            target_values=target_values,
            class_probability_matrix=this_probability_matrix,
            cost_function=cost_function, num_replicates=num_bootstrap_reps
        )
        this_cost_matrix = numpy.reshape(
            this_cost_array, (1, len(this_cost_array))
        )

        unpermuted_predictor_names.append(predictor_names[k])
        unpermuted_cost_matrix = numpy.concatenate(
            (unpermuted_cost_matrix, this_cost_matrix), axis=0
        )

        if numpy.mean(this_cost_array) > numpy.mean(best_cost_array):
            continue

        best_predictor_index = k + 0
        best_cost_array = this_cost_array + 0.

    if len(unpermuted_predictor_names) == 0:
        return None

    k = best_predictor_index + 0
    best_predictor_name = predictor_names[k]
    permuted_channel_flags[k] = False

    predictor_matrix = _unpermute_one_channel(
        predictor_matrix=predictor_matrix,
        clean_predictor_matrix=clean_predictor_matrix, channel_index=k
    )

    print('Best predictor = "{0:s}" ... cost = {1:.4f}'.format(
        best_predictor_name, numpy.mean(best_cost_array)
    ))

    return {
        PREDICTOR_MATRIX_KEY: predictor_matrix,
        PERMUTED_FLAGS_KEY: permuted_channel_flags,
        UNPERMUTED_PREDICTORS_KEY: unpermuted_predictor_names,
        UNPERMUTED_COST_MATRIX_KEY: unpermuted_cost_matrix,
        BEST_PREDICTOR_KEY: best_predictor_name,
        BEST_COST_ARRAY_KEY: best_cost_array
    }


def _check_input_args(clean_predictor_matrix, target_values, predictor_names,
                      num_bootstrap_reps):
    """Error-checks input args for forward or backwards permutation test.

    :param clean_predictor_matrix: Same as input `predictor_matrix` to
        `_run_forward_test_one_step` but clean (with no values permuted).
    :param target_values: See doc for `_run_forward_test_one_step`.
    :param predictor_names: Same.
    :param num_bootstrap_reps: Same.
    :return: num_bootstrap_reps: Integer version of input.
    """

    assert numpy.issubdtype(target_values.dtype, int)
    assert len(target_values.shape) == 1
    assert numpy.all(target_values >= 0)
    assert numpy.all(target_values <= 1)  # Assumes binary classification.

    assert isinstance(predictor_names, list)
    for n in predictor_names:
        assert isinstance(n, str)

    assert len(clean_predictor_matrix.shape) >= 4

    num_examples = len(target_values)
    num_channels = len(predictor_names)
    expected_dim = numpy.array(
        (num_examples,) + clean_predictor_matrix.shape[1:-1] + (num_channels,),
        dtype=int
    )
    assert numpy.array_equal(
        numpy.array(clean_predictor_matrix.shape, dtype=int),
        expected_dim
    )

    num_bootstrap_reps = int(numpy.round(num_bootstrap_reps))
    assert num_bootstrap_reps >= 1

    return num_bootstrap_reps


def negative_auc_function(target_values, class_probability_matrix):
    """Computes negative AUC (area under the ROC curve).

    This function works only for binary classification!

    :param target_values: length-E numpy array of target values (integers in
        0...1).
    :param class_probability_matrix: E-by-2 numpy array of predicted
        probabilities.
    :return: negative_auc: Negative AUC.
    :raises: TypeError: if `class_probability_matrix` contains more than 2
        classes.
    """

    num_classes = class_probability_matrix.shape[-1]

    if num_classes != 2:
        error_string = (
            'This function works only for binary classification, not '
            '{0:d}-class classification.'
        ).format(num_classes)

        raise TypeError(error_string)

    return -1 * sklearn_auc(
        y_true=target_values, y_score=class_probability_matrix[:, -1]
    )


def run_forward_test(
        model_object, clean_predictor_matrix, target_values, predictor_names,
        cost_function=negative_auc_function,
        num_bootstrap_reps=DEFAULT_NUM_BOOTSTRAP_REPS):
    """Runs single- and multi-pass versions of forward permutation test.

    C = number of channels (predictors)
    B = number of bootstrap replicates

    :param model_object: See doc for `_run_forward_test_one_step`.
    :param clean_predictor_matrix: Same as input `predictor_matrix` to
        `_run_forward_test_one_step` but clean (with no values permuted).
    :param target_values: See doc for `_run_forward_test_one_step`.
    :param predictor_names: Same.
    :param cost_function: Same.
    :param num_bootstrap_reps: Same.

    :return: result_dict: Dictionary with the following keys.
    result_dict["best_predictor_names"]: length-C list of best predictors.
        The [j]th element is the name of the [j]th predictor to be permanently
        permuted.
    result_dict["best_cost_matrix"]: C-by-B numpy array of costs after
        permutation.
    result_dict["original_cost_array"]: length-B numpy array of costs
        before permutation.
    result_dict["step1_predictor_names"]: length-C list of predictors in
        the order that they were permuted in step 1 (the single-pass test).
    result_dict["step1_cost_matrix"]: C-by-B numpy array of costs after
        permutation in step 1 (the single-pass test).
    result_dict["backwards_test"]: Boolean flag (always False).
    """

    num_bootstrap_reps = _check_input_args(
        clean_predictor_matrix=clean_predictor_matrix,
        target_values=target_values, predictor_names=predictor_names,
        num_bootstrap_reps=num_bootstrap_reps)

    num_examples = clean_predictor_matrix.shape[0]
    num_channels = clean_predictor_matrix.shape[-1]

    # Find original cost (before permutation).
    these_probs = model_object.predict(
        clean_predictor_matrix, batch_size=num_examples
    )
    class_probability_matrix = _event_probs_to_multiclass(these_probs)

    original_cost_array = _bootstrap_cost(
        target_values=target_values,
        class_probability_matrix=class_probability_matrix,
        cost_function=cost_function, num_replicates=num_bootstrap_reps)

    # Run the permutation test.
    step_num = 0
    predictor_matrix = clean_predictor_matrix + 0.
    permuted_channel_flags = numpy.full(num_channels, False, dtype=bool)

    step1_predictor_names = None
    step1_cost_matrix = None
    best_predictor_names = []
    best_cost_matrix = numpy.full((0, num_bootstrap_reps), numpy.nan)

    while True:
        print(SEPARATOR_STRING)
        step_num += 1

        this_dict = _run_forward_test_one_step(
            model_object=model_object, predictor_matrix=predictor_matrix,
            predictor_names=predictor_names, target_values=target_values,
            cost_function=cost_function, num_bootstrap_reps=num_bootstrap_reps,
            step_num=step_num, permuted_channel_flags=permuted_channel_flags
        )

        if this_dict is None:
            break

        predictor_matrix = this_dict[PREDICTOR_MATRIX_KEY]
        permuted_channel_flags = this_dict[PERMUTED_FLAGS_KEY]
        best_predictor_names.append(this_dict[BEST_PREDICTOR_KEY])

        this_best_cost_array = this_dict[BEST_COST_ARRAY_KEY]
        this_best_cost_matrix = numpy.reshape(
            this_best_cost_array, (1, len(this_best_cost_array))
        )
        best_cost_matrix = numpy.concatenate(
            (best_cost_matrix, this_best_cost_matrix), axis=0
        )

        if step_num == 1:
            step1_predictor_names = this_dict[PERMUTED_PREDICTORS_KEY]
            step1_cost_matrix = this_dict[PERMUTED_COST_MATRIX_KEY]

    return {
        BEST_PREDICTORS_KEY: best_predictor_names,
        BEST_COST_MATRIX_KEY: best_cost_matrix,
        ORIGINAL_COST_ARRAY_KEY: original_cost_array,
        STEP1_PREDICTORS_KEY: step1_predictor_names,
        STEP1_COST_MATRIX_KEY: step1_cost_matrix,
        BACKWARDS_FLAG: False
    }


def run_backwards_test(
        model_object, clean_predictor_matrix, target_values, predictor_names,
        cost_function=negative_auc_function,
        num_bootstrap_reps=DEFAULT_NUM_BOOTSTRAP_REPS):
    """Runs single- and multi-pass versions of backwards permutation test.

    C = number of channels (predictors)
    B = number of bootstrap replicates

    :param model_object: See doc for `run_forward_test`.
    :param clean_predictor_matrix: Same.
    :param target_values: Same.
    :param predictor_names: Same.
    :param cost_function: Same.
    :param num_bootstrap_reps: Same.

    :return: result_dict: Dictionary with the following keys.
    result_dict["best_predictor_names"]: length-C list of best predictors.
        The [j]th element is the name of the [j]th predictor to be permanently
        unpermuted.
    result_dict["best_cost_matrix"]: C-by-B numpy array of costs after
        unpermutation.
    result_dict["original_cost_array"]: length-B numpy array of costs
        before unpermutation.
    result_dict["step1_predictor_names"]: length-C list of predictors in
        the order that they were unpermuted in step 1 (the single-pass test).
    result_dict["step1_cost_matrix"]: C-by-B numpy array of costs after
        unpermutation in step 1 (the single-pass test).
    result_dict["backwards_test"]: Boolean flag (always True).
    """

    num_bootstrap_reps = _check_input_args(
        clean_predictor_matrix=clean_predictor_matrix,
        target_values=target_values, predictor_names=predictor_names,
        num_bootstrap_reps=num_bootstrap_reps)

    num_examples = clean_predictor_matrix.shape[0]
    num_channels = clean_predictor_matrix.shape[-1]

    # Permute all predictors.
    predictor_matrix = clean_predictor_matrix + 0.

    for k in range(num_channels):
        predictor_matrix = _permute_one_channel(
            predictor_matrix=predictor_matrix, channel_index=k
        )[0]

    # Find original cost (before unpermutation).
    these_probs = model_object.predict(
        predictor_matrix, batch_size=num_examples
    )
    class_probability_matrix = _event_probs_to_multiclass(these_probs)

    original_cost_array = _bootstrap_cost(
        target_values=target_values,
        class_probability_matrix=class_probability_matrix,
        cost_function=cost_function, num_replicates=num_bootstrap_reps)

    # Run the test.
    step_num = 0
    permuted_channel_flags = numpy.full(num_channels, True, dtype=bool)

    step1_predictor_names = None
    step1_cost_matrix = None
    best_predictor_names = []
    best_cost_matrix = numpy.full((0, num_bootstrap_reps), numpy.nan)

    while True:
        print(SEPARATOR_STRING)
        step_num += 1

        this_dict = _run_backwards_test_one_step(
            model_object=model_object, predictor_matrix=predictor_matrix,
            clean_predictor_matrix=clean_predictor_matrix,
            predictor_names=predictor_names, target_values=target_values,
            cost_function=cost_function, num_bootstrap_reps=num_bootstrap_reps,
            step_num=step_num, permuted_channel_flags=permuted_channel_flags
        )

        if this_dict is None:
            break

        predictor_matrix = this_dict[PREDICTOR_MATRIX_KEY]
        permuted_channel_flags = this_dict[PERMUTED_FLAGS_KEY]
        best_predictor_names.append(this_dict[BEST_PREDICTOR_KEY])

        this_best_cost_array = this_dict[BEST_COST_ARRAY_KEY]
        this_best_cost_matrix = numpy.reshape(
            this_best_cost_array, (1, len(this_best_cost_array))
        )
        best_cost_matrix = numpy.concatenate(
            (best_cost_matrix, this_best_cost_matrix), axis=0
        )

        if step_num == 1:
            step1_predictor_names = this_dict[UNPERMUTED_PREDICTORS_KEY]
            step1_cost_matrix = this_dict[UNPERMUTED_COST_MATRIX_KEY]

    return {
        BEST_PREDICTORS_KEY: best_predictor_names,
        BEST_COST_MATRIX_KEY: best_cost_matrix,
        ORIGINAL_COST_ARRAY_KEY: original_cost_array,
        STEP1_PREDICTORS_KEY: step1_predictor_names,
        STEP1_COST_MATRIX_KEY: step1_cost_matrix,
        BACKWARDS_FLAG: True
    }


def _get_limits_for_error_bar(cost_matrix, confidence_level):
    """Returns limits for error bar.

    S = number of steps in permutation test
    B = number of bootstrap replicates

    :param cost_matrix: S-by-B numpy array of costs.
    :param confidence_level: Confidence level for error bar (in range 0...1).
    :return: error_matrix: 2-by-S numpy array, where the first row contains
        negative errors and second row contains positive errors.
    """

    assert confidence_level >= 0.9
    assert confidence_level < 1.

    mean_costs = numpy.mean(cost_matrix, axis=-1)
    min_costs = numpy.percentile(
        cost_matrix, 50 * (1. - confidence_level), axis=-1
    )
    max_costs = numpy.percentile(
        cost_matrix, 50 * (1. + confidence_level), axis=-1
    )

    negative_errors = mean_costs - min_costs
    positive_errors = max_costs - mean_costs

    negative_errors = numpy.reshape(negative_errors, (1, negative_errors.size))
    positive_errors = numpy.reshape(positive_errors, (1, positive_errors.size))
    return numpy.vstack((negative_errors, positive_errors))


def _label_bar_graph(axes_object, label_strings, y_coords):
    """Labels bar graph with results of permutation test.

    B = number of bars

    :param axes_object: Will plot on these axes (instance of
        `matplotlib.axes._subplots.AxesSubplot`).
    :param label_strings: length-B list of labels (strings).
    :param y_coords: length-B numpy array with y-coordinate of each bar.
    """

    x_limits = axes_object.get_xlim()
    x_min = x_limits[0]
    x_max = x_limits[1]
    x_coord_for_text = x_min + 0.025 * (x_max - x_min)

    num_predictors = len(label_strings)

    for k in range(num_predictors):
        axes_object.text(
            x_coord_for_text, y_coords[k], '   ' + label_strings[k],
            color=BAR_TEXT_COLOUR, horizontalalignment='left',
            verticalalignment='center', fontsize=BAR_FONT_SIZE
        )


def _plot_bar_graph(
        cost_matrix, predictor_names, clean_cost_array,
        backwards_flag, multipass_flag, confidence_level, axes_object):
    """Plots bar graph with results of permutation test.

    C = number of channels (predictors)
    B = number of bootstrap replicates

    :param cost_matrix: (C + 1)-by-B numpy array of costs.  The first row
        contains costs at the beginning of the test -- i.e., before
        (un)permuting any predictors.
    :param predictor_names: length-C list of predictor names (strings).
    :param clean_cost_array: length-B numpy array of costs with clean predictors
        (no permutation).
    :param backwards_flag: Boolean flag.  If True, plotting results for the
        backwards test.  If False, for the forward test.
    :param multipass_flag: Boolean flag.  If True, plotting results for the
        multi-pass test.  If False, for the single-pass test.
    :param confidence_level: Confidence level for error bars.
    :param axes_object: Will plot on these axes (instance of
        `matplotlib.axes._subplots.AxesSubplot`).
    :return: axes_object: See input doc.
    """

    mean_clean_cost = numpy.mean(clean_cost_array)

    if numpy.any(cost_matrix < 0):
        cost_matrix *= -1
        mean_clean_cost *= -1
        x_axis_label = 'Area under ROC curve (AUC)'
    else:
        x_axis_label = 'Cost'

    if backwards_flag:
        bar_labels = ['All permuted'] + predictor_names
    else:
        bar_labels = ['None permuted'] + predictor_names

    bar_y_coords = numpy.linspace(
        0, len(bar_labels) - 1, num=len(bar_labels), dtype=float
    )

    if multipass_flag:
        bar_y_coords = bar_y_coords[::-1]

    if axes_object is None:
        _, axes_object = pyplot.subplots(
            1, 1, figsize=(FIGURE_WIDTH_INCHES, FIGURE_HEIGHT_INCHES)
        )

    bar_face_colours = [DEFAULT_BAR_COLOUR] * len(predictor_names)
    bar_face_colours.insert(0, NO_PERMUTATION_COLOUR)

    mean_costs = numpy.mean(cost_matrix, axis=-1)
    num_bootstrap_reps = cost_matrix.shape[1]

    if num_bootstrap_reps > 1:
        error_matrix = _get_limits_for_error_bar(
            cost_matrix=cost_matrix, confidence_level=confidence_level
        )

        axes_object.barh(
            bar_y_coords, mean_costs, color=bar_face_colours,
            edgecolor=BAR_EDGE_COLOUR, linewidth=BAR_EDGE_WIDTH,
            xerr=error_matrix, ecolor=ERROR_BAR_COLOUR,
            capsize=ERROR_BAR_CAP_SIZE, error_kw=ERROR_BAR_DICT
        )
    else:
        axes_object.barh(
            bar_y_coords, mean_costs, color=bar_face_colours,
            edgecolor=BAR_EDGE_COLOUR, linewidth=BAR_EDGE_WIDTH
        )

    reference_x_coords = numpy.full(2, mean_clean_cost)
    reference_y_coords = numpy.array([
        numpy.min(bar_y_coords) - 0.75,
        numpy.max(bar_y_coords) + 0.75
    ])

    axes_object.plot(
        reference_x_coords, reference_y_coords, color=REFERENCE_LINE_COLOUR,
        linestyle='--', linewidth=REFERENCE_LINE_WIDTH
    )

    axes_object.set_yticks([], [])
    axes_object.set_xlabel(x_axis_label)

    if backwards_flag:
        axes_object.set_ylabel('Variable unpermuted')
    else:
        axes_object.set_ylabel('Variable permuted')

    _label_bar_graph(
        axes_object=axes_object, label_strings=bar_labels, y_coords=bar_y_coords
    )
    axes_object.set_ylim(
        numpy.min(bar_y_coords) - 0.75, numpy.max(bar_y_coords) + 0.75
    )

    return axes_object


def plot_single_pass_test(
        result_dict, axes_object=None, num_predictors_to_plot=None,
        confidence_level=DEFAULT_CONFIDENCE_LEVEL):
    """Plots single-pass version of forward or backwards permutation test.

    :param result_dict: Dictionary created by `run_forward_test` or
        `run_backwards_test`.
    :param axes_object: Will plot on these axes (instance of
        `matplotlib.axes._subplots.AxesSubplot`).  If None, will create new
        axes.
    :param num_predictors_to_plot: Will plot only the K most important
        predictors, where K = `num_predictors_to_plot`.  If you want to plot all
        predictors, leave this alone.
    :param confidence_level: Confidence level for error bars (in range 0...1).
    :return: axes_object: See input doc.
    """

    # Check input args.
    predictor_names = result_dict[STEP1_PREDICTORS_KEY]

    if num_predictors_to_plot is None:
        num_predictors_to_plot = len(predictor_names)

    num_predictors_to_plot = min([
        num_predictors_to_plot, len(predictor_names)
    ])

    num_predictors_to_plot = int(numpy.round(num_predictors_to_plot))
    assert num_predictors_to_plot > 0

    # Set up arguments for `_plot_bar_graph`.
    backwards_flag = result_dict[BACKWARDS_FLAG]
    perturbed_cost_matrix = result_dict[STEP1_COST_MATRIX_KEY]
    mean_perturbed_costs = numpy.mean(perturbed_cost_matrix, axis=-1)

    if backwards_flag:
        sort_indices = numpy.argsort(
            mean_perturbed_costs
        )[:num_predictors_to_plot][::-1]
    else:
        sort_indices = numpy.argsort(
            mean_perturbed_costs
        )[-num_predictors_to_plot:]

    perturbed_cost_matrix = perturbed_cost_matrix[sort_indices, :]
    predictor_names = [predictor_names[k] for k in sort_indices]

    original_cost_array = result_dict[ORIGINAL_COST_ARRAY_KEY]
    original_cost_matrix = numpy.reshape(
        original_cost_array, (1, original_cost_array.size)
    )
    cost_matrix = numpy.concatenate(
        (original_cost_matrix, perturbed_cost_matrix), axis=0
    )

    # Plot.
    if backwards_flag:
        clean_cost_array = result_dict[BEST_COST_MATRIX_KEY][-1, :]
    else:
        clean_cost_array = original_cost_array

    return _plot_bar_graph(
        cost_matrix=cost_matrix, predictor_names=predictor_names,
        clean_cost_array=clean_cost_array,
        backwards_flag=backwards_flag, multipass_flag=False,
        confidence_level=confidence_level, axes_object=axes_object
    )


def plot_multipass_test(
        result_dict, axes_object=None, num_predictors_to_plot=None,
        confidence_level=DEFAULT_CONFIDENCE_LEVEL):
    """Plots multi-pass version of forward or backwards permutation test.

    :param result_dict: See doc for `plot_single_pass_test`.
    :param axes_object: Same.
    :param num_predictors_to_plot: Same.
    :param confidence_level: Same.
    :return: axes_object: Same.
    """

    # Check input args.
    predictor_names = result_dict[BEST_PREDICTORS_KEY]

    if num_predictors_to_plot is None:
        num_predictors_to_plot = len(predictor_names)

    num_predictors_to_plot = min([
        num_predictors_to_plot, len(predictor_names)
    ])

    num_predictors_to_plot = int(numpy.round(num_predictors_to_plot))
    assert num_predictors_to_plot > 0

    # Set up arguments for `_plot_bar_graph`.
    backwards_flag = result_dict[BACKWARDS_FLAG]
    perturbed_cost_matrix = result_dict[BEST_COST_MATRIX_KEY]

    perturbed_cost_matrix = perturbed_cost_matrix[:num_predictors_to_plot, :]
    predictor_names = predictor_names[:num_predictors_to_plot]

    original_cost_array = result_dict[ORIGINAL_COST_ARRAY_KEY]
    original_cost_matrix = numpy.reshape(
        original_cost_array, (1, original_cost_array.size)
    )
    cost_matrix = numpy.concatenate(
        (original_cost_matrix, perturbed_cost_matrix), axis=0
    )

    # Plot.
    if backwards_flag:
        clean_cost_array = result_dict[BEST_COST_MATRIX_KEY][-1, :]
    else:
        clean_cost_array = original_cost_array

    return _plot_bar_graph(
        cost_matrix=cost_matrix, predictor_names=predictor_names,
        clean_cost_array=clean_cost_array,
        backwards_flag=backwards_flag, multipass_flag=True,
        confidence_level=confidence_level, axes_object=axes_object
    )
