"""Main notebook for model interpretation."""

import os.path
import numpy
import keras
from matplotlib import pyplot
from interpretation import utils, normalization, binarization
from interpretation import plotting
from interpretation import cnn, permutation, saliency, class_activation
from interpretation import backwards_optimization as backwards_opt

NOTEBOOK_DIR_NAME = os.path.dirname(__file__)
SHORT_COURSE_DIR_NAME = os.path.dirname(NOTEBOOK_DIR_NAME)
IMAGE_DIR_NAME = '{0:s}/data/track_data_ncar_ams_3km_nc_small'.format(
    SHORT_COURSE_DIR_NAME
)

SEPARATOR_STRING = '\n\n' + '*' * 50 + '\n\n'
MINOR_SEPARATOR_STRING = '\n\n' + '-' * 50 + '\n\n'


def __read_files_example1():
    """Finds and reads image files."""

    image_file_names = utils.find_many_image_files(
        first_date_string='20150701', last_date_string='20150731',
        image_dir_name=IMAGE_DIR_NAME
    )

    image_dict = utils.read_many_image_files(image_file_names)
    print(MINOR_SEPARATOR_STRING)

    print('Variables in dictionary are as follows:')
    for this_key in image_dict.keys():
        print(this_key)

    print('\nPredictor variables are as follows:')
    predictor_names = image_dict[utils.PREDICTOR_NAMES_KEY]
    for this_name in predictor_names:
        print(this_name)

    these_predictor_values = (
        image_dict[utils.PREDICTOR_MATRIX_KEY][0, :5, :5, 0]
    )
    print('\nSome values of "{0:s}" for first storm object:\n{1:s}'.format(
        predictor_names[0], str(these_predictor_values)
    ))

    these_target_values = image_dict[utils.TARGET_MATRIX_KEY][0, :5, :5]
    print('\nSome values of "{0:s}" for first storm object:\n{1:s}'.format(
        image_dict[utils.TARGET_NAME_KEY], str(these_predictor_values)
    ))


def __read_validation_data():
    """Reads validation data (and finds training data)."""

    training_file_names = utils.find_many_image_files(
        first_date_string='20100101', last_date_string='20141224',
        image_dir_name=IMAGE_DIR_NAME
    )

    validation_file_names = utils.find_many_image_files(
        first_date_string='20150101', last_date_string='20151231',
        image_dir_name=IMAGE_DIR_NAME
    )

    validation_image_dict = utils.read_many_image_files(validation_file_names)


def __plot_predictors_example1(validation_image_dict):
    """Plots predictors for first storm object with wind barbs.

    :param validation_image_dict: Dictionary created by
        `utils.read_many_image_files`.
    """

    predictor_matrix = validation_image_dict[utils.PREDICTOR_MATRIX_KEY][0, ...]
    predictor_names = validation_image_dict[utils.PREDICTOR_NAMES_KEY]

    temperature_matrix_kelvins = predictor_matrix[
        ..., predictor_names.index(utils.TEMPERATURE_NAME)
    ]
    min_temp_kelvins = numpy.percentile(temperature_matrix_kelvins, 1)
    max_temp_kelvins = numpy.percentile(temperature_matrix_kelvins, 99)

    plotting.plot_many_predictors_with_barbs(
        predictor_matrix=predictor_matrix, predictor_names=predictor_names,
        min_colour_temp_kelvins=min_temp_kelvins,
        max_colour_temp_kelvins=max_temp_kelvins
    )

    pyplot.show()


def __plot_predictors_example2(validation_image_dict):
    """Plots predictors for extreme storm object with wind barbs.

    :param validation_image_dict: Dictionary created by
        `utils.read_many_image_files`.
    """

    target_matrix_s01 = validation_image_dict[utils.TARGET_MATRIX_KEY]
    example_index = numpy.unravel_index(
        numpy.argmax(target_matrix_s01), target_matrix_s01.shape
    )[0]

    predictor_matrix = validation_image_dict[utils.PREDICTOR_MATRIX_KEY][
        example_index, ...
    ]
    predictor_names = validation_image_dict[utils.PREDICTOR_NAMES_KEY]

    temperature_matrix_kelvins = predictor_matrix[
        ..., predictor_names.index(utils.TEMPERATURE_NAME)
    ]
    min_temp_kelvins = numpy.percentile(temperature_matrix_kelvins, 1)
    max_temp_kelvins = numpy.percentile(temperature_matrix_kelvins, 99)

    plotting.plot_many_predictors_with_barbs(
        predictor_matrix=predictor_matrix, predictor_names=predictor_names,
        min_colour_temp_kelvins=min_temp_kelvins,
        max_colour_temp_kelvins=max_temp_kelvins
    )

    pyplot.show()


def __plot_predictors_example3(validation_image_dict):
    """Plots predictors for extreme storm object without wind barbs.

    :param validation_image_dict: Dictionary created by
        `utils.read_many_image_files`.
    """

    target_matrix_s01 = validation_image_dict[utils.TARGET_MATRIX_KEY]
    example_index = numpy.unravel_index(
        numpy.argmax(target_matrix_s01), target_matrix_s01.shape
    )[0]

    predictor_matrix = validation_image_dict[utils.PREDICTOR_MATRIX_KEY][
        example_index, ...
    ]
    predictor_names = validation_image_dict[utils.PREDICTOR_NAMES_KEY]

    temperature_matrix_kelvins = predictor_matrix[
        ..., predictor_names.index(utils.TEMPERATURE_NAME)
    ]
    min_temp_kelvins = numpy.percentile(temperature_matrix_kelvins, 1)
    max_temp_kelvins = numpy.percentile(temperature_matrix_kelvins, 99)

    wind_indices = numpy.array([
        predictor_names.index(utils.U_WIND_NAME),
        predictor_names.index(utils.V_WIND_NAME)
    ], dtype=int)

    max_speed_m_s01 = numpy.percentile(
        numpy.absolute(predictor_matrix[..., wind_indices]), 99
    )

    plotting.plot_many_predictors_sans_barbs(
        predictor_matrix=predictor_matrix, predictor_names=predictor_names,
        min_colour_temp_kelvins=min_temp_kelvins,
        max_colour_temp_kelvins=max_temp_kelvins,
        max_colour_wind_speed_m_s01=max_speed_m_s01
    )

    pyplot.show()


def __get_normalization_params(training_file_names):
    """Computes normalization parameters.

    :param training_file_names: 1-D list of paths to training files.
    """

    normalization_dict = normalization.get_image_normalization_params(
        training_file_names
    )


def __norm_denorm_example(training_file_names, normalization_dict):
    """Normalizes and denormalizes some data.

    :param training_file_names: 1-D list of paths to training files.
    :param normalization_dict: Dictionary created by
        `normalization.get_image_normalization_params`.
    """

    image_dict = utils.read_image_file(training_file_names[0])
    predictor_names = image_dict[utils.PREDICTOR_NAMES_KEY]
    these_predictor_values = (
        image_dict[utils.PREDICTOR_MATRIX_KEY][0, :5, :5, 0]
    )

    print('\nOriginal values of "{0:s}" for first storm object:\n{1:s}'.format(
        predictor_names[0], str(these_predictor_values)
    ))

    image_dict[utils.PREDICTOR_MATRIX_KEY], _ = normalization.normalize_images(
        predictor_matrix=image_dict[utils.PREDICTOR_MATRIX_KEY],
        predictor_names=predictor_names,
        normalization_dict=normalization_dict
    )

    these_predictor_values = (
        image_dict[utils.PREDICTOR_MATRIX_KEY][0, :5, :5, 0]
    )
    print((
        '\nNormalized values of "{0:s}" for first storm object:\n{1:s}'
    ).format(
        predictor_names[0], str(these_predictor_values)
    ))

    image_dict[utils.PREDICTOR_MATRIX_KEY] = normalization.denormalize_images(
        predictor_matrix=image_dict[utils.PREDICTOR_MATRIX_KEY],
        predictor_names=predictor_names,
        normalization_dict=normalization_dict
    )

    these_predictor_values = (
        image_dict[utils.PREDICTOR_MATRIX_KEY][0, :5, :5, 0]
    )
    print((
        '\nDenormalized values of "{0:s}" for first storm object:\n{1:s}'
    ).format(
        predictor_names[0], str(these_predictor_values)
    ))


def __get_binarization_threshold(training_file_names):
    """Computes binarization threshold for target variable.

    :param training_file_names: 1-D list of paths to training files.
    """

    binarization_threshold = binarization.get_binarization_threshold(
        image_file_names=training_file_names, percentile_level=90.
    )


def __binarization_example(training_file_names, binarization_threshold):
    """Binarizes target values for one file.

    :param training_file_names: 1-D list of paths to training files.
    :param binarization_threshold: Binarization threshold.
    """

    image_dict = utils.read_image_file(training_file_names[0])
    target_matrix_s_01 = image_dict[utils.TARGET_MATRIX_KEY]
    numpy.random.shuffle(target_matrix_s_01)

    max_target_values_s01 = numpy.max(target_matrix_s_01, axis=(1, 2))
    print((
        '\nSpatial maxima of "{0:s}" for the first few storm objects:\n{1:s}'
    ).format(
        image_dict[utils.TARGET_NAME_KEY], str(max_target_values_s01[:20])
    ))

    target_values = binarization.binarize_target_images(
        target_matrix=image_dict[utils.TARGET_MATRIX_KEY],
        binarization_threshold=binarization_threshold
    )

    print((
        '\nBinarized target values for the first few storm objects:\n{0:s}'
    ).format(
        str(target_values[:20])
    ))


def _get_dense_layer_dimensions(num_input_neurons, num_classes,
                                num_dense_layers):
    """Returns dimensions (num input and output neurons) for each dense layer.

    D = number of dense layers

    :param num_input_neurons: Number of input neurons (features created by
        flattening layer).
    :param num_classes: Number of output classes (possible values of target
        variable).
    :param num_dense_layers: Number of dense layers.
    :return: num_inputs_by_layer: length-D numpy array with number of input
        neurons by dense layer.
    :return: num_outputs_by_layer: length-D numpy array with number of output
        neurons by dense layer.
    """

    if num_classes == 2:
        num_output_neurons = 1
    else:
        num_output_neurons = num_classes + 0

    e_folding_param = (
        -1.0 * num_dense_layers /
        numpy.log(float(num_output_neurons) / num_input_neurons)
    )

    dense_layer_indices = numpy.linspace(
        0, num_dense_layers - 1, num=num_dense_layers, dtype=float
    )
    num_inputs_by_layer = num_input_neurons * numpy.exp(
        -dense_layer_indices / e_folding_param
    )
    num_inputs_by_layer = numpy.round(num_inputs_by_layer).astype(int)

    num_outputs_by_layer = numpy.concatenate((
        num_inputs_by_layer[1:],
        numpy.array([num_output_neurons], dtype=int)
    ))

    return num_inputs_by_layer, num_outputs_by_layer


def _setup_cnn(training_file_names):
    """Sets up CNN (creates architecture).

    :param training_file_names: 1-D list of paths to training files.
    """

    NUM_CONV_LAYER_SETS = 2
    NUM_CONV_LAYERS_PER_SET = 2
    FIRST_NUM_CONV_FILTERS = 32
    NUM_CONV_FILTER_ROWS = 3
    NUM_CONV_FILTER_COLUMNS = 3

    L1_WEIGHT = 0.
    L2_WEIGHT = 0.001
    SLOPE_FOR_RELU = 0.2
    NUM_DENSE_LAYERS = 3
    DENSE_LAYER_DROPOUT_FRACTION = 0.5

    regularizer_object = keras.regularizers.l1_l2(l1=L1_WEIGHT, l2=L2_WEIGHT)

    # Find grid dimensions and number of channels.
    image_dict = utils.read_image_file(training_file_names[0])
    predictor_matrix = image_dict[utils.PREDICTOR_MATRIX_KEY]
    num_grid_rows = predictor_matrix.shape[1]
    num_grid_columns = predictor_matrix.shape[2]
    num_channels = predictor_matrix.shape[3]

    # Create input layer.
    input_layer_object = keras.layers.Input(
        shape=(num_grid_rows, num_grid_columns, num_channels)
    )

    # Add convolutional and pooling layers, with activation and batch
    # normalization after each conv layer.
    current_num_filters = None
    current_layer_object = None

    for _ in range(NUM_CONV_LAYER_SETS):
        for _ in range(NUM_CONV_LAYERS_PER_SET):
            if current_num_filters is None:
                current_num_filters = FIRST_NUM_CONV_FILTERS + 0
                current_layer_object = input_layer_object
            else:
                current_num_filters *= 2

            current_layer_object = keras.layers.Conv2D(
                filters=current_num_filters,
                kernel_size=(NUM_CONV_FILTER_ROWS, NUM_CONV_FILTER_COLUMNS),
                strides=(1, 1), padding='valid', data_format='channels_last',
                dilation_rate=(1, 1), activation=None, use_bias=True,
                kernel_initializer='glorot_uniform', bias_initializer='zeros',
                kernel_regularizer=regularizer_object
            )(current_layer_object)

            current_layer_object = keras.layers.LeakyReLU(
                alpha=SLOPE_FOR_RELU
            )(current_layer_object)

            current_layer_object = keras.layers.BatchNormalization(
                axis=-1, center=True, scale=True
            )(current_layer_object)

        current_layer_object = keras.layers.MaxPooling2D(
            pool_size=(2, 2), strides=(2, 2),
            padding='valid', data_format='channels_last'
        )(current_layer_object)

    these_dimensions = numpy.array(
        current_layer_object.get_shape().as_list()[1:], dtype=int
    )
    num_scalar_features = numpy.prod(these_dimensions)

    current_layer_object = keras.layers.Flatten()(current_layer_object)

    # Add non-final dense layers.
    _, num_outputs_by_dense_layer = _get_dense_layer_dimensions(
        num_input_neurons=num_scalar_features, num_classes=2,
        num_dense_layers=NUM_DENSE_LAYERS
    )

    for k in range(NUM_DENSE_LAYERS - 1):
        current_layer_object = keras.layers.Dense(
            num_outputs_by_dense_layer[k], activation=None, use_bias=True,
            kernel_initializer='glorot_uniform', bias_initializer='zeros'
        )(current_layer_object)

        current_layer_object = keras.layers.LeakyReLU(
            alpha=SLOPE_FOR_RELU
        )(current_layer_object)

        current_layer_object = keras.layers.Dropout(
            rate=DENSE_LAYER_DROPOUT_FRACTION
        )(current_layer_object)

        current_layer_object = keras.layers.BatchNormalization(
            axis=-1, center=True, scale=True
        )(current_layer_object)

    # Add final dense layer (output layer).
    current_layer_object = keras.layers.Dense(
        1, activation=None, use_bias=True,
        kernel_initializer='glorot_uniform', bias_initializer='zeros',
    )(current_layer_object)

    current_layer_object = keras.layers.Activation(
        'sigmoid'
    )(current_layer_object)

    if NUM_DENSE_LAYERS == 1:
        current_layer_object = keras.layers.Dropout(
            rate=DENSE_LAYER_DROPOUT_FRACTION
        )(current_layer_object)

    # Compile model.
    cnn_model_object = keras.models.Model(
        inputs=input_layer_object, outputs=current_layer_object
    )
    cnn_model_object.compile(
        loss=keras.losses.binary_crossentropy,
        optimizer=keras.optimizers.Adam(),
        metrics=cnn.LIST_OF_METRIC_FUNCTIONS
    )
    cnn_model_object.summary()


def train_cnn(
        model_object, training_file_names, validation_file_names,
        num_examples_per_batch, normalization_dict, binarization_threshold,
        num_epochs, num_training_batches_per_epoch,
        num_validation_batches_per_epoch, output_file_name):
    """Trains CNN.

    :param model_object: Untrained model (instance of `keras.models.Model` or
        `keras.models.Sequential`).
    :param training_file_names: 1-D list of paths to training files (readable by
        `utils.read_image_file`).
    :param validation_file_names: Same but for validation files.
    :param num_examples_per_batch: See doc for `data_generator`.
    :param normalization_dict: Same.
    :param binarization_threshold: Same.
    :param num_epochs: Number of epochs.
    :param num_training_batches_per_epoch: Number of training batches per epoch.
    :param num_validation_batches_per_epoch: Number of validation batches per
        epoch.
    :param output_file_name: Path to output file.  The trained model will be
        saved here, after each epoch, as an HDF5 file.
    :return: cnn_metadata_dict: See doc for `write_metadata`.
    """

    utils.create_directory(file_name=output_file_name)

    cnn_metadata_dict = {
        cnn.TRAINING_FILES_KEY: training_file_names,
        cnn.NORMALIZATION_DICT_KEY: normalization_dict,
        cnn.BINARIZATION_THRESHOLD_KEY: binarization_threshold,
        cnn.NUM_EXAMPLES_PER_BATCH_KEY: num_examples_per_batch,
        cnn.NUM_TRAINING_BATCHES_KEY: num_training_batches_per_epoch,
        cnn.VALIDATION_FILES_KEY: validation_file_names,
        cnn.NUM_VALIDATION_BATCHES_KEY: num_validation_batches_per_epoch
    }

    checkpoint_object = keras.callbacks.ModelCheckpoint(
        filepath=output_file_name, monitor='val_loss',
        save_best_only=True, save_weights_only=False, mode='min', period=1,
        verbose=1
    )
    early_stopping_object = keras.callbacks.EarlyStopping(
        monitor='val_loss', min_delta=cnn.CROSS_ENTROPY_PATIENCE,
        patience=cnn.EARLY_STOPPING_PATIENCE_EPOCHS, mode='min', verbose=1
    )
    plateau_object = keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss', factor=cnn.PLATEAU_LEARNING_RATE_MULTIPLIER,
        patience=cnn.PLATEAU_PATIENCE_EPOCHS, mode='min',
        min_delta=cnn.CROSS_ENTROPY_PATIENCE,
        cooldown=cnn.PLATEAU_COOLDOWN_EPOCHS,
        verbose=1
    )

    list_of_callback_objects = [
        checkpoint_object, early_stopping_object, plateau_object
    ]

    training_generator = cnn.data_generator(
        image_file_names=training_file_names,
        num_examples_per_batch=num_examples_per_batch,
        normalization_dict=normalization_dict,
        binarization_threshold=binarization_threshold
    )

    validation_generator = cnn.data_generator(
        image_file_names=validation_file_names,
        num_examples_per_batch=num_examples_per_batch,
        normalization_dict=normalization_dict,
        binarization_threshold=binarization_threshold
    )

    model_object.fit_generator(
        generator=training_generator,
        steps_per_epoch=num_training_batches_per_epoch, epochs=num_epochs,
        verbose=1, callbacks=list_of_callback_objects, workers=0,
        validation_data=validation_generator,
        validation_steps=num_validation_batches_per_epoch
    )

    return cnn_metadata_dict


def __train_cnn_example(
        cnn_model_object, training_file_names, validation_file_names,
        normalization_dict, binarization_threshold):
    """Trains CNN.

    :param cnn_model_object: See doc for `train_cnn`.
    :param training_file_names: Same.
    :param validation_file_names: Same.
    :param normalization_dict: Same.
    :param binarization_threshold: Same.
    """

    output_file_name = '{0:s}/custom_cnn/model.h5'.format(IMAGE_DIR_NAME)

    # train_cnn(
    #     model_object=cnn_model_object,
    #     training_file_names=training_file_names,
    #     validation_file_names=validation_file_names,
    #     normalization_dict=normalization_dict,
    #     binarization_threshold=binarization_threshold,
    #     num_training_batches_per_epoch=32,
    #     num_validation_batches_per_epoch=16,
    #     num_examples_per_batch=1024, num_epochs=10,
    #     output_file_name=output_file_name
    # )

    train_cnn(
        model_object=cnn_model_object,
        training_file_names=training_file_names,
        validation_file_names=validation_file_names,
        normalization_dict=normalization_dict,
        binarization_threshold=binarization_threshold,
        num_training_batches_per_epoch=10,
        num_validation_batches_per_epoch=10,
        num_examples_per_batch=256, num_epochs=10,
        output_file_name=output_file_name
    )


def __run_forward_permutation_test(cnn_model_object, validation_image_dict,
                                   normalization_dict, binarization_threshold):
    """Runs forward version of permutation test.

    :param cnn_model_object: Trained CNN (instance of `keras.models.Model` or
        `keras.models.Sequential`).
    :param validation_image_dict: Dictionary created by
        `utils.read_many_image_files`.
    :param normalization_dict: Dictionary created by
        `normalization.get_image_normalization_params`.
    :param binarization_threshold: Binarization threshold for target variable.
    """

    target_values = binarization.binarize_target_images(
        target_matrix=validation_image_dict[utils.TARGET_MATRIX_KEY],
        binarization_threshold=binarization_threshold
    )
    clean_predictor_matrix = (
        validation_image_dict[utils.PREDICTOR_MATRIX_KEY] + 0.
    )
    clean_predictor_matrix, _ = normalization.normalize_images(
        predictor_matrix=clean_predictor_matrix,
        predictor_names=validation_image_dict[utils.PREDICTOR_NAMES_KEY],
        normalization_dict=normalization_dict
    )

    # permutation_result_dict = permutation.run_forward_test(
    #     model_object=cnn_model_object,
    #     clean_predictor_matrix=clean_predictor_matrix,
    #     target_values=target_values,
    #     predictor_names=validation_image_dict[utils.PREDICTOR_NAMES_KEY]
    # )

    permutation_result_dict = permutation.run_forward_test(
        model_object=cnn_model_object,
        clean_predictor_matrix=clean_predictor_matrix[:2000, ...],
        target_values=target_values[:2000],
        predictor_names=validation_image_dict[utils.PREDICTOR_NAMES_KEY]
    )


def __plot_forward_permutation_test(permutation_result_dict):
    """Plots forward version of permutation test.

    :param permutation_result_dict: Dictionary created by
        `permutation.run_forward_test`.
    """

    axes_object = permutation.plot_single_pass_test(
        result_dict=permutation_result_dict
    )
    axes_object.set_title('Single-pass forward test')

    axes_object = permutation.plot_multipass_test(
        result_dict=permutation_result_dict
    )
    axes_object.set_title('Multi-pass forward test')


def __run_backwards_permutation_test(
        cnn_model_object, validation_image_dict, normalization_dict,
        binarization_threshold):
    """Runs backwards version of permutation test.

    :param cnn_model_object: See doc for `__run_forward_permutation_test`.
    :param validation_image_dict: Same.
    :param normalization_dict: Same.
    :param binarization_threshold: Same.
    """

    target_values = binarization.binarize_target_images(
        target_matrix=validation_image_dict[utils.TARGET_MATRIX_KEY],
        binarization_threshold=binarization_threshold
    )
    clean_predictor_matrix = (
        validation_image_dict[utils.PREDICTOR_MATRIX_KEY] + 0.
    )
    clean_predictor_matrix, _ = normalization.normalize_images(
        predictor_matrix=clean_predictor_matrix,
        predictor_names=validation_image_dict[utils.PREDICTOR_NAMES_KEY],
        normalization_dict=normalization_dict
    )

    # permutation_result_dict = permutation.run_backwards_test(
    #     model_object=cnn_model_object,
    #     clean_predictor_matrix=clean_predictor_matrix,
    #     target_values=target_values,
    #     predictor_names=validation_image_dict[utils.PREDICTOR_NAMES_KEY]
    # )

    permutation_result_dict = permutation.run_backwards_test(
        model_object=cnn_model_object,
        clean_predictor_matrix=clean_predictor_matrix[:2000, ...],
        target_values=target_values[:2000],
        predictor_names=validation_image_dict[utils.PREDICTOR_NAMES_KEY]
    )


def __plot_backwards_permutation_test(permutation_result_dict):
    """Plots backwards version of permutation test.

    :param permutation_result_dict: Dictionary created by
        `permutation.run_backwards_test`.
    """

    axes_object = permutation.plot_single_pass_test(
        result_dict=permutation_result_dict
    )
    axes_object.set_title('Single-pass backwards test')

    axes_object = permutation.plot_multipass_test(
        result_dict=permutation_result_dict
    )
    axes_object.set_title('Multi-pass backwards test')


def __saliency_example1(cnn_model_object, validation_image_dict,
                        normalization_dict):
    """Computes saliency wrt strong-rotation probability for first storm object.

    :param cnn_model_object: Trained CNN (instance of `keras.models.Model` or
        `keras.models.Sequential`).
    :param validation_image_dict: Dictionary created by
        `utils.read_many_image_files`.
    :param normalization_dict: Dictionary created by
        `normalization.get_image_normalization_params`.
    """

    predictor_matrix_denorm = (
        validation_image_dict[utils.PREDICTOR_MATRIX_KEY][0, ...] + 0.
    )
    predictor_names = validation_image_dict[utils.PREDICTOR_NAMES_KEY]

    predictor_matrix_norm, _ = normalization.normalize_images(
        predictor_matrix=predictor_matrix_denorm + 0.,
        predictor_names=predictor_names,
        normalization_dict=normalization_dict
    )
    predictor_matrix_norm = numpy.expand_dims(predictor_matrix_norm, axis=0)

    saliency_matrix = saliency.get_saliency_maps_for_class(
        model_object=cnn_model_object, target_class=1,
        list_of_input_matrices=[predictor_matrix_norm]
    )[0][0, ...]

    saliency_matrix = saliency.smooth_saliency_maps(
        saliency_matrices=[saliency_matrix], smoothing_radius_grid_cells=1
    )[0]

    temperature_matrix_kelvins = predictor_matrix_denorm[
        ..., predictor_names.index(utils.TEMPERATURE_NAME)
    ]
    min_temp_kelvins = numpy.percentile(temperature_matrix_kelvins, 1)
    max_temp_kelvins = numpy.percentile(temperature_matrix_kelvins, 99)

    wind_indices = numpy.array([
        predictor_names.index(utils.U_WIND_NAME),
        predictor_names.index(utils.V_WIND_NAME)
    ], dtype=int)

    max_speed_m_s01 = numpy.percentile(
        numpy.absolute(predictor_matrix_denorm[..., wind_indices]), 99
    )

    figure_object, axes_object_matrix = (
        plotting.plot_many_predictors_sans_barbs(
            predictor_matrix=predictor_matrix_denorm,
            predictor_names=predictor_names,
            min_colour_temp_kelvins=min_temp_kelvins,
            max_colour_temp_kelvins=max_temp_kelvins,
            max_colour_wind_speed_m_s01=max_speed_m_s01)
    )

    max_contour_value = numpy.percentile(numpy.absolute(saliency_matrix), 99)

    saliency.plot_saliency_maps(
        saliency_matrix_3d=saliency_matrix,
        axes_object_matrix=axes_object_matrix,
        colour_map_object=pyplot.get_cmap('Greys'),
        max_contour_value=max_contour_value,
        contour_interval=max_contour_value / 8
    )


def __saliency_example2(cnn_model_object, validation_image_dict,
                        normalization_dict):
    """Computes saliency wrt weak-rotation probability for first storm object.

    :param cnn_model_object: See doc for `__saliency_example1`.
    :param validation_image_dict: Same.
    :param normalization_dict: Same.
    """

    predictor_matrix_denorm = (
        validation_image_dict[utils.PREDICTOR_MATRIX_KEY][0, ...] + 0.
    )
    predictor_names = validation_image_dict[utils.PREDICTOR_NAMES_KEY]

    predictor_matrix_norm, _ = normalization.normalize_images(
        predictor_matrix=predictor_matrix_denorm + 0.,
        predictor_names=predictor_names,
        normalization_dict=normalization_dict
    )
    predictor_matrix_norm = numpy.expand_dims(predictor_matrix_norm, axis=0)

    saliency_matrix = saliency.get_saliency_maps_for_class(
        model_object=cnn_model_object, target_class=0,
        list_of_input_matrices=[predictor_matrix_norm]
    )[0][0, ...]

    saliency_matrix = saliency.smooth_saliency_maps(
        saliency_matrices=[saliency_matrix], smoothing_radius_grid_cells=1
    )[0]

    temperature_matrix_kelvins = predictor_matrix_denorm[
        ..., predictor_names.index(utils.TEMPERATURE_NAME)
    ]
    min_temp_kelvins = numpy.percentile(temperature_matrix_kelvins, 1)
    max_temp_kelvins = numpy.percentile(temperature_matrix_kelvins, 99)

    wind_indices = numpy.array([
        predictor_names.index(utils.U_WIND_NAME),
        predictor_names.index(utils.V_WIND_NAME)
    ], dtype=int)

    max_speed_m_s01 = numpy.percentile(
        numpy.absolute(predictor_matrix_denorm[..., wind_indices]), 99
    )

    figure_object, axes_object_matrix = (
        plotting.plot_many_predictors_sans_barbs(
            predictor_matrix=predictor_matrix_denorm,
            predictor_names=predictor_names,
            min_colour_temp_kelvins=min_temp_kelvins,
            max_colour_temp_kelvins=max_temp_kelvins,
            max_colour_wind_speed_m_s01=max_speed_m_s01)
    )

    max_contour_value = numpy.percentile(numpy.absolute(saliency_matrix), 99)

    saliency.plot_saliency_maps(
        saliency_matrix_3d=saliency_matrix,
        axes_object_matrix=axes_object_matrix,
        colour_map_object=pyplot.get_cmap('Greys'),
        max_contour_value=max_contour_value,
        contour_interval=max_contour_value / 8
    )


def __saliency_example3(cnn_model_object, validation_image_dict,
                        normalization_dict):
    """Computes saliency wrt strong-rotation probability for extreme storm obj.

    :param cnn_model_object: See doc for `__saliency_example1`.
    :param validation_image_dict: Same.
    :param normalization_dict: Same.
    """

    target_matrix_s01 = validation_image_dict[utils.TARGET_MATRIX_KEY]
    example_index = numpy.unravel_index(
        numpy.argmax(target_matrix_s01), target_matrix_s01.shape
    )[0]

    predictor_matrix_denorm = (
        validation_image_dict[utils.PREDICTOR_MATRIX_KEY][example_index, ...]
        + 0.
    )
    predictor_names = validation_image_dict[utils.PREDICTOR_NAMES_KEY]

    predictor_matrix_norm, _ = normalization.normalize_images(
        predictor_matrix=predictor_matrix_denorm + 0.,
        predictor_names=predictor_names,
        normalization_dict=normalization_dict
    )
    predictor_matrix_norm = numpy.expand_dims(predictor_matrix_norm, axis=0)

    saliency_matrix = saliency.get_saliency_maps_for_class(
        model_object=cnn_model_object, target_class=1,
        list_of_input_matrices=[predictor_matrix_norm]
    )[0][0, ...]

    saliency_matrix = saliency.smooth_saliency_maps(
        saliency_matrices=[saliency_matrix], smoothing_radius_grid_cells=1
    )[0]

    temperature_matrix_kelvins = predictor_matrix_denorm[
        ..., predictor_names.index(utils.TEMPERATURE_NAME)
    ]
    min_temp_kelvins = numpy.percentile(temperature_matrix_kelvins, 1)
    max_temp_kelvins = numpy.percentile(temperature_matrix_kelvins, 99)

    wind_indices = numpy.array([
        predictor_names.index(utils.U_WIND_NAME),
        predictor_names.index(utils.V_WIND_NAME)
    ], dtype=int)

    max_speed_m_s01 = numpy.percentile(
        numpy.absolute(predictor_matrix_denorm[..., wind_indices]), 99
    )

    figure_object, axes_object_matrix = (
        plotting.plot_many_predictors_sans_barbs(
            predictor_matrix=predictor_matrix_denorm,
            predictor_names=predictor_names,
            min_colour_temp_kelvins=min_temp_kelvins,
            max_colour_temp_kelvins=max_temp_kelvins,
            max_colour_wind_speed_m_s01=max_speed_m_s01)
    )

    max_contour_value = numpy.percentile(numpy.absolute(saliency_matrix), 99)

    saliency.plot_saliency_maps(
        saliency_matrix_3d=saliency_matrix,
        axes_object_matrix=axes_object_matrix,
        colour_map_object=pyplot.get_cmap('Greys'),
        max_contour_value=max_contour_value,
        contour_interval=max_contour_value / 8
    )


def __saliency_example4(cnn_model_object, validation_image_dict,
                        normalization_dict):
    """Computes saliency wrt weak-rotation probability for extreme storm object.

    :param cnn_model_object: See doc for `__saliency_example1`.
    :param validation_image_dict: Same.
    :param normalization_dict: Same.
    """

    target_matrix_s01 = validation_image_dict[utils.TARGET_MATRIX_KEY]
    example_index = numpy.unravel_index(
        numpy.argmax(target_matrix_s01), target_matrix_s01.shape
    )[0]

    predictor_matrix_denorm = (
        validation_image_dict[utils.PREDICTOR_MATRIX_KEY][example_index, ...]
        + 0.
    )
    predictor_names = validation_image_dict[utils.PREDICTOR_NAMES_KEY]

    predictor_matrix_norm, _ = normalization.normalize_images(
        predictor_matrix=predictor_matrix_denorm + 0.,
        predictor_names=predictor_names,
        normalization_dict=normalization_dict
    )
    predictor_matrix_norm = numpy.expand_dims(predictor_matrix_norm, axis=0)

    saliency_matrix = saliency.get_saliency_maps_for_class(
        model_object=cnn_model_object, target_class=0,
        list_of_input_matrices=[predictor_matrix_norm]
    )[0][0, ...]

    saliency_matrix = saliency.smooth_saliency_maps(
        saliency_matrices=[saliency_matrix], smoothing_radius_grid_cells=1
    )[0]

    temperature_matrix_kelvins = predictor_matrix_denorm[
        ..., predictor_names.index(utils.TEMPERATURE_NAME)
    ]
    min_temp_kelvins = numpy.percentile(temperature_matrix_kelvins, 1)
    max_temp_kelvins = numpy.percentile(temperature_matrix_kelvins, 99)

    wind_indices = numpy.array([
        predictor_names.index(utils.U_WIND_NAME),
        predictor_names.index(utils.V_WIND_NAME)
    ], dtype=int)

    max_speed_m_s01 = numpy.percentile(
        numpy.absolute(predictor_matrix_denorm[..., wind_indices]), 99
    )

    figure_object, axes_object_matrix = (
        plotting.plot_many_predictors_sans_barbs(
            predictor_matrix=predictor_matrix_denorm,
            predictor_names=predictor_names,
            min_colour_temp_kelvins=min_temp_kelvins,
            max_colour_temp_kelvins=max_temp_kelvins,
            max_colour_wind_speed_m_s01=max_speed_m_s01)
    )

    max_contour_value = numpy.percentile(numpy.absolute(saliency_matrix), 99)

    saliency.plot_saliency_maps(
        saliency_matrix_3d=saliency_matrix,
        axes_object_matrix=axes_object_matrix,
        colour_map_object=pyplot.get_cmap('Greys'),
        max_contour_value=max_contour_value,
        contour_interval=max_contour_value / 8
    )


def __gradcam_example1(cnn_model_object, validation_image_dict,
                       normalization_dict):
    """Computes positive-class activation for first storm object.

    :param cnn_model_object: See doc for `__saliency_example1`.
    :param validation_image_dict: Same.
    :param normalization_dict: Same.
    """

    predictor_matrix_denorm = (
        validation_image_dict[utils.PREDICTOR_MATRIX_KEY][0, ...] + 0.
    )
    predictor_names = validation_image_dict[utils.PREDICTOR_NAMES_KEY]

    predictor_matrix_norm, _ = normalization.normalize_images(
        predictor_matrix=predictor_matrix_denorm + 0.,
        predictor_names=predictor_names,
        normalization_dict=normalization_dict
    )

    temperature_matrix_kelvins = predictor_matrix_denorm[
        ..., predictor_names.index(utils.TEMPERATURE_NAME)
    ]
    min_temp_kelvins = numpy.percentile(temperature_matrix_kelvins, 1)
    max_temp_kelvins = numpy.percentile(temperature_matrix_kelvins, 99)

    wind_indices = numpy.array([
        predictor_names.index(utils.U_WIND_NAME),
        predictor_names.index(utils.V_WIND_NAME)
    ], dtype=int)

    max_speed_m_s01 = numpy.percentile(
        numpy.absolute(predictor_matrix_denorm[..., wind_indices]), 99
    )

    target_layer_names = [
        'batch_normalization_1', 'batch_normalization_2',
        'batch_normalization_3', 'batch_normalization_4'
    ]

    for this_layer_name in target_layer_names:
        class_activation_matrix = class_activation.run_gradcam(
            model_object=cnn_model_object, input_matrix=predictor_matrix_norm,
            target_class=1, target_layer_name=this_layer_name
        )

        figure_object, axes_object_matrix = (
            plotting.plot_many_predictors_sans_barbs(
                predictor_matrix=predictor_matrix_denorm,
                predictor_names=predictor_names,
                min_colour_temp_kelvins=min_temp_kelvins,
                max_colour_temp_kelvins=max_temp_kelvins,
                max_colour_wind_speed_m_s01=max_speed_m_s01)
        )

        max_contour_value = numpy.percentile(class_activation_matrix, 99)

        class_activation.plot_2d_cam(
            class_activation_matrix_2d=class_activation_matrix,
            axes_object_matrix=axes_object_matrix,
            num_channels=predictor_matrix_norm.shape[-1],
            colour_map_object=pyplot.get_cmap('Greys'),
            min_contour_value=max_contour_value / 15,
            max_contour_value=max_contour_value,
            contour_interval=max_contour_value / 15
        )

        figure_object.suptitle(
            'CAM for layer "{0:s}"'.format(this_layer_name)
        )
        pyplot.show()


def __gradcam_example2(cnn_model_object, validation_image_dict,
                       normalization_dict):
    """Computes negative-class activation for first storm object.

    :param cnn_model_object: See doc for `__saliency_example1`.
    :param validation_image_dict: Same.
    :param normalization_dict: Same.
    """

    predictor_matrix_denorm = (
        validation_image_dict[utils.PREDICTOR_MATRIX_KEY][0, ...] + 0.
    )
    predictor_names = validation_image_dict[utils.PREDICTOR_NAMES_KEY]

    predictor_matrix_norm, _ = normalization.normalize_images(
        predictor_matrix=predictor_matrix_denorm + 0.,
        predictor_names=predictor_names,
        normalization_dict=normalization_dict
    )

    temperature_matrix_kelvins = predictor_matrix_denorm[
        ..., predictor_names.index(utils.TEMPERATURE_NAME)
    ]
    min_temp_kelvins = numpy.percentile(temperature_matrix_kelvins, 1)
    max_temp_kelvins = numpy.percentile(temperature_matrix_kelvins, 99)

    wind_indices = numpy.array([
        predictor_names.index(utils.U_WIND_NAME),
        predictor_names.index(utils.V_WIND_NAME)
    ], dtype=int)

    max_speed_m_s01 = numpy.percentile(
        numpy.absolute(predictor_matrix_denorm[..., wind_indices]), 99
    )

    target_layer_names = [
        'batch_normalization_1', 'batch_normalization_2',
        'batch_normalization_3', 'batch_normalization_4'
    ]

    for this_layer_name in target_layer_names:
        class_activation_matrix = class_activation.run_gradcam(
            model_object=cnn_model_object, input_matrix=predictor_matrix_norm,
            target_class=0, target_layer_name=this_layer_name
        )

        figure_object, axes_object_matrix = (
            plotting.plot_many_predictors_sans_barbs(
                predictor_matrix=predictor_matrix_denorm,
                predictor_names=predictor_names,
                min_colour_temp_kelvins=min_temp_kelvins,
                max_colour_temp_kelvins=max_temp_kelvins,
                max_colour_wind_speed_m_s01=max_speed_m_s01)
        )

        max_contour_value = numpy.percentile(class_activation_matrix, 99)

        class_activation.plot_2d_cam(
            class_activation_matrix_2d=class_activation_matrix,
            axes_object_matrix=axes_object_matrix,
            num_channels=predictor_matrix_norm.shape[-1],
            colour_map_object=pyplot.get_cmap('Greys'),
            min_contour_value=max_contour_value / 15,
            max_contour_value=max_contour_value,
            contour_interval=max_contour_value / 15
        )

        figure_object.suptitle(
            'CAM for layer "{0:s}"'.format(this_layer_name)
        )
        pyplot.show()


def __gradcam_example3(cnn_model_object, validation_image_dict,
                       normalization_dict):
    """Computes positive-class activation for extreme storm object.

    :param cnn_model_object: See doc for `__saliency_example1`.
    :param validation_image_dict: Same.
    :param normalization_dict: Same.
    """

    target_matrix_s01 = validation_image_dict[utils.TARGET_MATRIX_KEY]
    example_index = numpy.unravel_index(
        numpy.argmax(target_matrix_s01), target_matrix_s01.shape
    )[0]

    predictor_matrix_denorm = (
        validation_image_dict[utils.PREDICTOR_MATRIX_KEY][example_index, ...]
        + 0.
    )
    predictor_names = validation_image_dict[utils.PREDICTOR_NAMES_KEY]

    predictor_matrix_norm, _ = normalization.normalize_images(
        predictor_matrix=predictor_matrix_denorm + 0.,
        predictor_names=predictor_names,
        normalization_dict=normalization_dict
    )

    temperature_matrix_kelvins = predictor_matrix_denorm[
        ..., predictor_names.index(utils.TEMPERATURE_NAME)
    ]
    min_temp_kelvins = numpy.percentile(temperature_matrix_kelvins, 1)
    max_temp_kelvins = numpy.percentile(temperature_matrix_kelvins, 99)

    wind_indices = numpy.array([
        predictor_names.index(utils.U_WIND_NAME),
        predictor_names.index(utils.V_WIND_NAME)
    ], dtype=int)

    max_speed_m_s01 = numpy.percentile(
        numpy.absolute(predictor_matrix_denorm[..., wind_indices]), 99
    )

    target_layer_names = [
        'batch_normalization_1', 'batch_normalization_2',
        'batch_normalization_3', 'batch_normalization_4'
    ]

    for this_layer_name in target_layer_names:
        class_activation_matrix = class_activation.run_gradcam(
            model_object=cnn_model_object, input_matrix=predictor_matrix_norm,
            target_class=1, target_layer_name=this_layer_name
        )

        figure_object, axes_object_matrix = (
            plotting.plot_many_predictors_sans_barbs(
                predictor_matrix=predictor_matrix_denorm,
                predictor_names=predictor_names,
                min_colour_temp_kelvins=min_temp_kelvins,
                max_colour_temp_kelvins=max_temp_kelvins,
                max_colour_wind_speed_m_s01=max_speed_m_s01)
        )

        max_contour_value = numpy.percentile(class_activation_matrix, 99)

        class_activation.plot_2d_cam(
            class_activation_matrix_2d=class_activation_matrix,
            axes_object_matrix=axes_object_matrix,
            num_channels=predictor_matrix_norm.shape[-1],
            colour_map_object=pyplot.get_cmap('Greys'),
            min_contour_value=max_contour_value / 15,
            max_contour_value=max_contour_value,
            contour_interval=max_contour_value / 15
        )

        figure_object.suptitle(
            'CAM for layer "{0:s}"'.format(this_layer_name)
        )
        pyplot.show()


def __backwards_opt_example1(cnn_model_object, validation_image_dict,
                             normalization_dict):
    """Optimizes first storm object for positive class.

    :param cnn_model_object: See doc for `__saliency_example1`.
    :param validation_image_dict: Same.
    :param normalization_dict: Same.
    """

    orig_predictor_matrix_denorm = (
        validation_image_dict[utils.PREDICTOR_MATRIX_KEY][0, ...] + 0.
    )
    predictor_names = validation_image_dict[utils.PREDICTOR_NAMES_KEY]

    orig_predictor_matrix_norm, _ = normalization.normalize_images(
        predictor_matrix=orig_predictor_matrix_denorm + 0.,
        predictor_names=predictor_names,
        normalization_dict=normalization_dict
    )
    orig_predictor_matrix_norm = numpy.expand_dims(
        orig_predictor_matrix_norm, axis=0
    )

    new_predictor_matrix_norm = backwards_opt.optimize_example_for_class(
        model_object=cnn_model_object, input_matrix=orig_predictor_matrix_norm,
        target_class=1, num_iterations=1000, learning_rate=0.0001,
        l2_weight=0.0005
    )[0][0, ...]

    new_predictor_matrix_denorm = normalization.denormalize_images(
        predictor_matrix=new_predictor_matrix_norm,
        predictor_names=predictor_names,
        normalization_dict=normalization_dict
    )

    temperature_index = predictor_names.index(utils.TEMPERATURE_NAME)
    combined_temp_matrix_kelvins = numpy.concatenate((
        orig_predictor_matrix_denorm[..., temperature_index],
        new_predictor_matrix_denorm[..., temperature_index]
    ), axis=0)

    min_temp_kelvins = numpy.percentile(combined_temp_matrix_kelvins, 1)
    max_temp_kelvins = numpy.percentile(combined_temp_matrix_kelvins, 99)

    figure_object, _ = plotting.plot_many_predictors_with_barbs(
        predictor_matrix=orig_predictor_matrix_denorm,
        predictor_names=predictor_names,
        min_colour_temp_kelvins=min_temp_kelvins,
        max_colour_temp_kelvins=max_temp_kelvins
    )

    figure_object.suptitle('Real example (before optimization)')
    pyplot.show()

    figure_object, _ = plotting.plot_many_predictors_with_barbs(
        predictor_matrix=new_predictor_matrix_denorm,
        predictor_names=predictor_names,
        min_colour_temp_kelvins=min_temp_kelvins,
        max_colour_temp_kelvins=max_temp_kelvins
    )

    figure_object.suptitle('Synthetic example (after optimization)')
    pyplot.show()


def __backwards_opt_example2(cnn_model_object, validation_image_dict,
                             normalization_dict):
    """Optimizes first storm object for negative class.

    :param cnn_model_object: See doc for `__saliency_example1`.
    :param validation_image_dict: Same.
    :param normalization_dict: Same.
    """

    orig_predictor_matrix_denorm = (
        validation_image_dict[utils.PREDICTOR_MATRIX_KEY][0, ...] + 0.
    )
    predictor_names = validation_image_dict[utils.PREDICTOR_NAMES_KEY]

    orig_predictor_matrix_norm, _ = normalization.normalize_images(
        predictor_matrix=orig_predictor_matrix_denorm + 0.,
        predictor_names=predictor_names,
        normalization_dict=normalization_dict
    )
    orig_predictor_matrix_norm = numpy.expand_dims(
        orig_predictor_matrix_norm, axis=0
    )

    new_predictor_matrix_norm = backwards_opt.optimize_example_for_class(
        model_object=cnn_model_object, input_matrix=orig_predictor_matrix_norm,
        target_class=0, num_iterations=1000, learning_rate=0.0001,
        l2_weight=0.
    )[0][0, ...]

    new_predictor_matrix_denorm = normalization.denormalize_images(
        predictor_matrix=new_predictor_matrix_norm,
        predictor_names=predictor_names,
        normalization_dict=normalization_dict
    )

    temperature_index = predictor_names.index(utils.TEMPERATURE_NAME)
    combined_temp_matrix_kelvins = numpy.concatenate((
        orig_predictor_matrix_denorm[..., temperature_index],
        new_predictor_matrix_denorm[..., temperature_index]
    ), axis=0)

    min_temp_kelvins = numpy.percentile(combined_temp_matrix_kelvins, 1)
    max_temp_kelvins = numpy.percentile(combined_temp_matrix_kelvins, 99)

    figure_object, _ = plotting.plot_many_predictors_with_barbs(
        predictor_matrix=orig_predictor_matrix_denorm,
        predictor_names=predictor_names,
        min_colour_temp_kelvins=min_temp_kelvins,
        max_colour_temp_kelvins=max_temp_kelvins
    )

    figure_object.suptitle('Real example (before optimization)')
    pyplot.show()

    figure_object, _ = plotting.plot_many_predictors_with_barbs(
        predictor_matrix=new_predictor_matrix_denorm,
        predictor_names=predictor_names,
        min_colour_temp_kelvins=min_temp_kelvins,
        max_colour_temp_kelvins=max_temp_kelvins
    )

    figure_object.suptitle('Synthetic example (after optimization)')
    pyplot.show()


def __backwards_opt_example3(cnn_model_object, validation_image_dict,
                             normalization_dict):
    """Optimizes extreme storm object for negative class.

    :param cnn_model_object: See doc for `__saliency_example1`.
    :param validation_image_dict: Same.
    :param normalization_dict: Same.
    """

    target_matrix_s01 = validation_image_dict[utils.TARGET_MATRIX_KEY]
    example_index = numpy.unravel_index(
        numpy.argmax(target_matrix_s01), target_matrix_s01.shape
    )[0]

    orig_predictor_matrix_denorm = (
        validation_image_dict[utils.PREDICTOR_MATRIX_KEY][example_index, ...]
        + 0.
    )
    predictor_names = validation_image_dict[utils.PREDICTOR_NAMES_KEY]

    orig_predictor_matrix_norm, _ = normalization.normalize_images(
        predictor_matrix=orig_predictor_matrix_denorm + 0.,
        predictor_names=predictor_names,
        normalization_dict=normalization_dict
    )
    orig_predictor_matrix_norm = numpy.expand_dims(
        orig_predictor_matrix_norm, axis=0
    )

    new_predictor_matrix_norm = backwards_opt.optimize_example_for_class(
        model_object=cnn_model_object, input_matrix=orig_predictor_matrix_norm,
        target_class=0, num_iterations=1000, learning_rate=0.001,
        l2_weight=0.
    )[0][0, ...]

    new_predictor_matrix_denorm = normalization.denormalize_images(
        predictor_matrix=new_predictor_matrix_norm,
        predictor_names=predictor_names,
        normalization_dict=normalization_dict
    )

    temperature_index = predictor_names.index(utils.TEMPERATURE_NAME)
    combined_temp_matrix_kelvins = numpy.concatenate((
        orig_predictor_matrix_denorm[..., temperature_index],
        new_predictor_matrix_denorm[..., temperature_index]
    ), axis=0)

    min_temp_kelvins = numpy.percentile(combined_temp_matrix_kelvins, 1)
    max_temp_kelvins = numpy.percentile(combined_temp_matrix_kelvins, 99)

    figure_object, _ = plotting.plot_many_predictors_with_barbs(
        predictor_matrix=orig_predictor_matrix_denorm,
        predictor_names=predictor_names,
        min_colour_temp_kelvins=min_temp_kelvins,
        max_colour_temp_kelvins=max_temp_kelvins
    )

    figure_object.suptitle('Real example (before optimization)')
    pyplot.show()

    figure_object, _ = plotting.plot_many_predictors_with_barbs(
        predictor_matrix=new_predictor_matrix_denorm,
        predictor_names=predictor_names,
        min_colour_temp_kelvins=min_temp_kelvins,
        max_colour_temp_kelvins=max_temp_kelvins
    )

    figure_object.suptitle('Synthetic example (after optimization)')
    pyplot.show()


def __backwards_opt_example4(cnn_model_object, validation_image_dict,
                             normalization_dict):
    """Optimizes extreme storm object for negative class.

    :param cnn_model_object: See doc for `__saliency_example1`.
    :param validation_image_dict: Same.
    :param normalization_dict: Same.
    """

    target_matrix_s01 = validation_image_dict[utils.TARGET_MATRIX_KEY]
    example_index = numpy.unravel_index(
        numpy.argmax(target_matrix_s01), target_matrix_s01.shape
    )[0]

    orig_predictor_matrix_denorm = (
        validation_image_dict[utils.PREDICTOR_MATRIX_KEY][example_index, ...]
        + 0.
    )
    predictor_names = validation_image_dict[utils.PREDICTOR_NAMES_KEY]

    orig_predictor_matrix_norm, _ = normalization.normalize_images(
        predictor_matrix=orig_predictor_matrix_denorm + 0.,
        predictor_names=predictor_names,
        normalization_dict=normalization_dict
    )
    orig_predictor_matrix_norm = numpy.expand_dims(
        orig_predictor_matrix_norm, axis=0
    )

    new_predictor_matrix_norm = backwards_opt.optimize_example_for_class(
        model_object=cnn_model_object, input_matrix=orig_predictor_matrix_norm,
        target_class=0, num_iterations=1000, learning_rate=2.5e-4,
        l2_weight=2.5e-5
    )[0][0, ...]

    new_predictor_matrix_denorm = normalization.denormalize_images(
        predictor_matrix=new_predictor_matrix_norm,
        predictor_names=predictor_names,
        normalization_dict=normalization_dict
    )

    temperature_index = predictor_names.index(utils.TEMPERATURE_NAME)
    combined_temp_matrix_kelvins = numpy.concatenate((
        orig_predictor_matrix_denorm[..., temperature_index],
        new_predictor_matrix_denorm[..., temperature_index]
    ), axis=0)

    min_temp_kelvins = numpy.percentile(combined_temp_matrix_kelvins, 1)
    max_temp_kelvins = numpy.percentile(combined_temp_matrix_kelvins, 99)

    figure_object, _ = plotting.plot_many_predictors_with_barbs(
        predictor_matrix=orig_predictor_matrix_denorm,
        predictor_names=predictor_names,
        min_colour_temp_kelvins=min_temp_kelvins,
        max_colour_temp_kelvins=max_temp_kelvins
    )

    figure_object.suptitle('Real example (before optimization)')
    pyplot.show()

    figure_object, _ = plotting.plot_many_predictors_with_barbs(
        predictor_matrix=new_predictor_matrix_denorm,
        predictor_names=predictor_names,
        min_colour_temp_kelvins=min_temp_kelvins,
        max_colour_temp_kelvins=max_temp_kelvins
    )

    figure_object.suptitle('Synthetic example (after optimization)')
    pyplot.show()
