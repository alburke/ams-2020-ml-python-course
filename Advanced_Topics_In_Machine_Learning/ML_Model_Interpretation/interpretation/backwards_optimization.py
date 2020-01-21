"""Helper methods for backwards optimization."""

import numpy
from keras import backend as K

DEFAULT_LEARNING_RATE = 0.001
DEFAULT_NUM_ITERATIONS = 1000
DEFAULT_L2_WEIGHT = 1.


def _optimize_input_one_example(
        model_object, input_matrix, activation_tensor, loss_tensor,
        num_iterations, learning_rate, l2_weight):
    """Optimizes inputs (predictors) for one example.

    :param model_object: See doc for `optimize_example_for_class`.
    :param input_matrix: Same.
    :param activation_tensor: Keras tensor defining activation of relevant model
        component.
    :param loss_tensor: Keras tensor defining loss (difference between actual
        and desired activation).
    :param num_iterations: See doc for `optimize_example_for_class`.
    :param learning_rate: Same.
    :param l2_weight: Same.
    :return: optimized_input_matrix: Same.
    :return: initial_activation: Same.
    :return: final_activation: Same.
    """

    if isinstance(model_object.input, list):
        input_tensor = model_object.input[0]
    else:
        input_tensor = model_object.input

    optimized_input_matrix = input_matrix + 0.

    if l2_weight is not None:
        difference_tensor = (
            input_tensor[0, ...] - optimized_input_matrix[0, ...]
        )

        loss_tensor += l2_weight * K.sum(difference_tensor ** 2)

    gradient_tensor = K.gradients(loss_tensor, [input_tensor])[0]
    gradient_tensor /= K.maximum(
        K.sqrt(K.mean(gradient_tensor ** 2)),
        K.epsilon()
    )

    grad_descent_function = K.function(
        [input_tensor, K.learning_phase()],
        [activation_tensor, loss_tensor, gradient_tensor]
    )

    initial_activation = None
    current_loss = None
    current_activation = None

    for j in range(num_iterations):
        vals = grad_descent_function([optimized_input_matrix, 0])
        current_loss = vals[1]
        current_activation = vals[0][0]
        current_gradient = vals[2]

        if j == 0:
            initial_activation = current_activation

        if numpy.mod(j, 100) == 0:
            print((
                'Loss after {0:d} of {1:d} iterations = {2:.2e} ... '
                'activation = {3:.2e}'
            ).format(
                j, num_iterations, current_loss, current_activation
            ))

        optimized_input_matrix -= current_gradient * learning_rate

    final_activation = current_activation

    print((
        'Loss after {0:d} iterations = {1:.2e} ... activation = {2:.2e}'
    ).format(
        num_iterations, current_loss, final_activation
    ))

    return optimized_input_matrix, initial_activation, final_activation


def optimize_example_for_class(
        model_object, input_matrix, target_class,
        num_iterations=DEFAULT_NUM_ITERATIONS,
        learning_rate=DEFAULT_LEARNING_RATE,
        l2_weight=DEFAULT_L2_WEIGHT):
    """Optimizes one example to maximize probability of target class.

    :param model_object: Trained model (instance of `keras.models.Model` or
        `keras.models.Sequential`).
    :param input_matrix: numpy array with inputs (predictors) for one example.
    :param target_class: Target class.  Must be an integer in 0...(K - 1), where
        K = number of classes.
    :param num_iterations: Number of iterations for gradient descent.
    :param learning_rate: Learning rate for gradient descent.
    :param l2_weight: Strength of L_2 penalty (on difference between original
        and optimized input matrices).  If you do not want an L_2 penalty, make
        this None.
    :return: optimized_input_matrix: Same as input matrix but with different
        values.
    :return: initial_activation: Initial activation of relevant model component
        (before any backwards optimization).
    :return: final_activation: Final activation (after backwards optimization).
    """

    # Check input args.
    target_class = int(numpy.round(target_class))
    num_iterations = int(numpy.round(num_iterations))

    assert not numpy.any(numpy.isnan(input_matrix))
    assert target_class >= 0
    assert num_iterations > 0
    assert learning_rate > 0.
    if l2_weight <= 0:
        l2_weight = None

    num_output_neurons = (
        model_object.layers[-1].output.get_shape().as_list()[-1]
    )

    if num_output_neurons == 1:
        assert target_class <= 1

        activation_tensor = model_object.layers[-1].output[..., 0]

        if target_class == 1:
            loss_tensor = K.mean((activation_tensor - 1) ** 2)
        else:
            loss_tensor = K.mean(activation_tensor ** 2)
    else:
        assert target_class < num_output_neurons

        activation_tensor = model_object.layers[-1].output[..., target_class]
        loss_tensor = K.mean((activation_tensor - 1) ** 2)

    return _optimize_input_one_example(
        model_object=model_object, input_matrix=input_matrix,
        activation_tensor=activation_tensor, loss_tensor=loss_tensor,
        num_iterations=num_iterations, learning_rate=learning_rate,
        l2_weight=l2_weight
    )
