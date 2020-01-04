"""Helper methods for CNNs (convolutional neural nets)."""

import copy
import json
import random
import os.path
import numpy
import keras.models
from interpretation import utils, normalization, binarization
from evaluation import keras_metrics

PLATEAU_PATIENCE_EPOCHS = 3
PLATEAU_LEARNING_RATE_MULTIPLIER = 0.5
PLATEAU_COOLDOWN_EPOCHS = 0
EARLY_STOPPING_PATIENCE_EPOCHS = 15
CROSS_ENTROPY_PATIENCE = 0.005

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

TRAINING_FILES_KEY = 'training_file_names'
NORMALIZATION_DICT_KEY = 'normalization_dict'
BINARIZATION_THRESHOLD_KEY = 'binarization_threshold'
NUM_EXAMPLES_PER_BATCH_KEY = 'num_examples_per_batch'
NUM_TRAINING_BATCHES_KEY = 'num_training_batches_per_epoch'
VALIDATION_FILES_KEY = 'validation_file_names'
NUM_VALIDATION_BATCHES_KEY = 'num_validation_batches_per_epoch'


def _metadata_numpy_to_list(cnn_metadata_dict):
    """Converts numpy arrays in model metadata to lists.

    This is needed so that the metadata can be written to a JSON file (JSON does
    not handle numpy arrays).

    This method does not overwrite the original dictionary.

    :param cnn_metadata_dict: See doc for `write_metadata`.
    :return: new_metadata_dict: Same but with lists instead of numpy arrays.
    """

    new_metadata_dict = copy.deepcopy(cnn_metadata_dict)

    if NORMALIZATION_DICT_KEY in new_metadata_dict.keys():
        this_norm_dict = new_metadata_dict[NORMALIZATION_DICT_KEY]

        for this_key in this_norm_dict.keys():
            if isinstance(this_norm_dict[this_key], numpy.ndarray):
                this_norm_dict[this_key] = this_norm_dict[this_key].tolist()

    return new_metadata_dict


def _metadata_list_to_numpy(cnn_metadata_dict):
    """Converts lists in model metadata to numpy arrays.

    This method is the inverse of `_metadata_numpy_to_list`.

    This method overwrites the original dictionary.

    :param cnn_metadata_dict: See doc for `write_metadata`.
    :return: cnn_metadata_dict: Same but with numpy arrays instead of lists.
    """

    if NORMALIZATION_DICT_KEY in cnn_metadata_dict.keys():
        this_norm_dict = cnn_metadata_dict[NORMALIZATION_DICT_KEY]

        for this_key in this_norm_dict.keys():
            this_norm_dict[this_key] = numpy.array(this_norm_dict[this_key])

    return cnn_metadata_dict


def data_generator(image_file_names, num_examples_per_batch, normalization_dict,
                   binarization_threshold):
    """Generates training or validation examples on the fly.

    E = number of examples (storm objects)
    M = number of rows in each storm-centered grid
    N = number of columns in each storm-centered grid
    C = number of channels (predictor variables)

    :param image_file_names: 1-D list of paths to input files (readable by
        `utils.read_image_file`).
    :param num_examples_per_batch: Number of examples per training batch.
    :param normalization_dict: Dictionary with params used to normalize
        predictors.  See doc for `normalization.normalize_images`.
    :param binarization_threshold: Threshold used to binarize target variable.
        See doc for `binarization.binarize_target_images`.
    :return: predictor_matrix: E-by-M-by-N-by-C numpy array of predictor values.
    :return: target_values: length-E numpy array of target values (integers in
        0...1).
    :raises: TypeError: if `normalization_dict is None`.
    """

    # TODO(thunderhoser): Maybe add upsampling or downsampling.

    if normalization_dict is None:
        error_string = 'normalization_dict cannot be None.  Must be specified.'
        raise TypeError(error_string)

    random.shuffle(image_file_names)
    num_files = len(image_file_names)
    file_index = 0

    num_examples_in_memory = 0
    full_predictor_matrix = None
    full_target_matrix = None
    predictor_names = None

    while True:
        while num_examples_in_memory < num_examples_per_batch:
            print('Reading data from: "{0:s}"...'.format(
                image_file_names[file_index]
            ))

            this_image_dict = utils.read_image_file(image_file_names[file_index])
            predictor_names = this_image_dict[utils.PREDICTOR_NAMES_KEY]

            file_index += 1
            if file_index >= num_files:
                file_index = 0

            if full_target_matrix is None or full_target_matrix.size == 0:
                full_predictor_matrix = (
                    this_image_dict[utils.PREDICTOR_MATRIX_KEY] + 0.
                )
                full_target_matrix = this_image_dict[utils.TARGET_MATRIX_KEY] + 0.

            else:
                full_predictor_matrix = numpy.concatenate((
                    full_predictor_matrix,
                    this_image_dict[utils.PREDICTOR_MATRIX_KEY]
                ), axis=0)

                full_target_matrix = numpy.concatenate((
                    full_target_matrix, this_image_dict[utils.TARGET_MATRIX_KEY]
                ), axis=0)

            num_examples_in_memory = full_target_matrix.shape[0]

        batch_indices = numpy.linspace(
            0, num_examples_in_memory - 1, num=num_examples_in_memory,
            dtype=int
        )
        batch_indices = numpy.random.choice(
            batch_indices, size=num_examples_per_batch, replace=False
        )

        predictor_matrix, _ = normalization.normalize_images(
            predictor_matrix=full_predictor_matrix[batch_indices, ...],
            predictor_names=predictor_names,
            normalization_dict=normalization_dict
        )

        target_values = binarization.binarize_target_images(
            target_matrix=full_target_matrix[batch_indices, ...],
            binarization_threshold=binarization_threshold
        )

        print('Fraction of examples in positive class: {0:.4f}'.format(
            numpy.mean(target_values)
        ))

        num_examples_in_memory = 0
        full_predictor_matrix = None
        full_target_matrix = None

        yield (predictor_matrix.astype('float32'), target_values)


def train_model(
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
        TRAINING_FILES_KEY: training_file_names,
        NORMALIZATION_DICT_KEY: normalization_dict,
        BINARIZATION_THRESHOLD_KEY: binarization_threshold,
        NUM_EXAMPLES_PER_BATCH_KEY: num_examples_per_batch,
        NUM_TRAINING_BATCHES_KEY: num_training_batches_per_epoch,
        VALIDATION_FILES_KEY: validation_file_names,
        NUM_VALIDATION_BATCHES_KEY: num_validation_batches_per_epoch
    }

    checkpoint_object = keras.callbacks.ModelCheckpoint(
        filepath=output_file_name, monitor='val_loss',
        save_best_only=True, save_weights_only=False, mode='min', period=1,
        verbose=1
    )
    early_stopping_object = keras.callbacks.EarlyStopping(
        monitor='val_loss', min_delta=CROSS_ENTROPY_PATIENCE,
        patience=EARLY_STOPPING_PATIENCE_EPOCHS, mode='min', verbose=1
    )
    plateau_object = keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss', factor=PLATEAU_LEARNING_RATE_MULTIPLIER,
        patience=PLATEAU_PATIENCE_EPOCHS, mode='min',
        min_delta=CROSS_ENTROPY_PATIENCE, cooldown=PLATEAU_COOLDOWN_EPOCHS,
        verbose=1
    )

    list_of_callback_objects = [
        checkpoint_object, early_stopping_object, plateau_object
    ]

    training_generator = data_generator(
        image_file_names=training_file_names,
        num_examples_per_batch=num_examples_per_batch,
        normalization_dict=normalization_dict,
        binarization_threshold=binarization_threshold
    )

    validation_generator = data_generator(
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


def read_model(hdf5_file_name):
    """Reads model from HDF5 file.

    :param hdf5_file_name: Path to input file.
    :return: model_object: Instance of `keras.models.Model`.
    """

    return keras.models.load_model(
        hdf5_file_name, custom_objects=METRIC_FUNCTION_DICT
    )


def apply_cnn(model_object, predictor_matrix, verbose=True,
              output_layer_name=None):
    """Applies trained CNN to new data.

    E = number of examples (storm objects)
    M = number of rows in each storm-centered grid
    N = number of columns in each storm-centered grid
    C = number of channels (predictor variables)

    :param model_object: Trained model (instance of `keras.models.Model` or
        `keras.models.Sequential`).
    :param predictor_matrix: E-by-M-by-N-by-C numpy array of predictor values.
    :param verbose: Boolean flag.  If True, will print progress messages.
    :param output_layer_name: Name of output layer.  If None, will use the
        actual output layer and return predicted probabilities.  If specified,
        will return "features" (outputs from the given layer).

    If `output_layer_name is None`...

    :return: forecast_probabilities: length-E numpy array with probabilities of
        positive class.

    If `output_layer_name` is specified...

    :return: feature_matrix: numpy array of features from the given layer.
        There is no guarantee on the shape of this array, except that the first
        axis has length E.
    """

    num_examples = predictor_matrix.shape[0]
    num_examples_per_batch = 1000

    if output_layer_name is None:
        model_object_to_use = model_object
    else:
        model_object_to_use = keras.models.Model(
            inputs=model_object.input,
            outputs=model_object.get_layer(name=output_layer_name).output
        )

    output_array = None

    for i in range(0, num_examples, num_examples_per_batch):
        this_first_index = i
        this_last_index = min(
            [i + num_examples_per_batch - 1, num_examples - 1]
        )

        if verbose:
            print('Applying model to examples {0:d}-{1:d} of {2:d}...'.format(
                this_first_index, this_last_index, num_examples
            ))

        these_indices = numpy.linspace(
            this_first_index, this_last_index,
            num=this_last_index - this_first_index + 1, dtype=int
        )

        this_output_array = model_object_to_use.predict(
            predictor_matrix[these_indices, ...],
            batch_size=num_examples_per_batch
        )

        if output_layer_name is None:
            this_output_array = this_output_array[:, -1]

        if output_array is None:
            output_array = this_output_array + 0.
        else:
            output_array = numpy.concatenate(
                (output_array, this_output_array), axis=0
            )

    if verbose:
        print('Have applied model to all {0:d} examples!'.format(num_examples))

    return output_array


def apply_upconvnet(
        cnn_model_object, predictor_matrix, cnn_feature_layer_name,
        upconvnet_model_object, verbose=True):
    """Applies trained upconvnet to new data.

    :param cnn_model_object: See doc for `apply_cnn`.
    :param predictor_matrix: Same.
    :param cnn_feature_layer_name: See doc for `output_layer_name` in
        `apply_cnn`.
    :param upconvnet_model_object: Trained upconvnet (instance of
        `keras.models.Model` or `keras.models.Sequential`).  The input to the
        upconvnet is the output from `cnn_feature_layer_name` in the CNN.
    :param verbose: Boolean flag.  If True, will print progress messages.
    :return: recon_predictor_matrix: Reconstructed version of input.
    """

    num_examples = predictor_matrix.shape[0]
    num_examples_per_batch = 1000
    recon_predictor_matrix = numpy.full(predictor_matrix.shape, numpy.nan)

    for i in range(0, num_examples, num_examples_per_batch):
        this_first_index = i
        this_last_index = min(
            [i + num_examples_per_batch - 1, num_examples - 1]
        )

        if verbose:
            print((
                'Using upconvnet to reconstruct examples {0:d}-{1:d} of '
                '{2:d}...'
            ).format(
                this_first_index, this_last_index, num_examples
            ))

        these_indices = numpy.linspace(
            this_first_index, this_last_index,
            num=this_last_index - this_first_index + 1, dtype=int
        )

        this_feature_matrix = apply_cnn(
            model_object=cnn_model_object,
            predictor_matrix=predictor_matrix[these_indices, ...],
            output_layer_name=cnn_feature_layer_name, verbose=False
        )

        recon_predictor_matrix[these_indices, ...] = (
            upconvnet_model_object.predict(
                this_feature_matrix, batch_size=len(these_indices)
            )
        )

    if verbose:
        print('Have used upconvnet to reconstruct all {0:d} examples!'.format(
            num_examples
        ))

    return recon_predictor_matrix


def find_model_metafile(model_file_name, raise_error_if_missing=False):
    """Finds metafile for CNN.

    :param model_file_name: Path to file with CNN.
    :param raise_error_if_missing: Boolean flag.  If True and metafile is not
        found, this method will error out.
    :return: model_metafile_name: Path to file with metadata.  If file is not
        found and `raise_error_if_missing = False`, this will be the expected
        path.
    :raises: ValueError: if metafile is not found and
        `raise_error_if_missing = True`.
    """

    model_directory_name, pathless_model_file_name = os.path.split(
        model_file_name)
    model_metafile_name = '{0:s}/{1:s}_metadata.json'.format(
        model_directory_name, os.path.splitext(pathless_model_file_name)[0]
    )

    if not os.path.isfile(model_metafile_name) and raise_error_if_missing:
        error_string = 'Cannot find file.  Expected at: "{0:s}"'.format(
            model_metafile_name)
        raise ValueError(error_string)

    return model_metafile_name


def write_metadata(cnn_metadata_dict, json_file_name):
    """Writes metadata for CNN to JSON file.

    :param cnn_metadata_dict: Dictionary with the following keys.
    cnn_metadata_dict['training_file_names']: 1-D list of paths to training
        files (readable by `utils.read_image_file`).
    cnn_metadata_dict['normalization_dict']: See doc for
        `normalization.normalize_images`.
    cnn_metadata_dict['binarization_threshold']: Threshold used to binarize
        target variable.
    cnn_metadata_dict['num_examples_per_batch']: Number of examples per batch.
    cnn_metadata_dict['num_training_batches_per_epoch']: Number of training
        batches per epoch.
    cnn_metadata_dict['validation_file_names']: 1-D list of paths to validation
        files (readable by `utils.read_image_file`).
    cnn_metadata_dict['num_validation_batches_per_epoch']: Number of validation
        batches per epoch.

    :param json_file_name: Path to output file.
    """

    utils.create_directory(file_name=json_file_name)

    new_metadata_dict = _metadata_numpy_to_list(cnn_metadata_dict)
    with open(json_file_name, 'w') as this_file:
        json.dump(new_metadata_dict, this_file)


def read_metadata(json_file_name):
    """Reads metadata for CNN from JSON file.

    :param json_file_name: Path to output file.
    :return: cnn_metadata_dict: See doc for `write_metadata`.
    """

    with open(json_file_name) as this_file:
        cnn_metadata_dict = json.load(this_file)
        return _metadata_list_to_numpy(cnn_metadata_dict)
