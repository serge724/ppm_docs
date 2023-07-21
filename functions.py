# Author: Sergej Levich
# Journal article: Sergej Levich et al., Decision Support Systems, https://doi.org/10.1016/j.dss.2023.114043

import os
import pickle
import numpy as np
import pandas as pd
import tensorflow as tf

# configure tensorflow dataset for selected context and task
def configure_tf_dataset(tf_dataset, max_instance_length, max_page_n, n_states, n_types, context, target, batch_size, drop_remainder, shuffle, **kwargs):

    # dataset without context
    if context == 'none':
        if shuffle: tf_dataset = tf_dataset.shuffle(len(tf_dataset), reshuffle_each_iteration = True) # apply (perfect) shuffling
        if target == 'type':
            tf_dataset = tf_dataset.map(lambda x, y, t: (
                (tf.sparse.to_dense(x)[0:-1,:], tf.sparse.to_dense(y)[0:-1,:]), # remove last element from the sequence (EOS) event to present the input
                (tf.sparse.to_dense(t)[1:,:], ) # remove first element from the sequence to present the target
            ))
            # one-hot encode output variable for classification branch
            tf_dataset = tf_dataset.map(lambda x, y: (x, (tf.one_hot(tf.squeeze(y[0]), n_types), )))
        if target == 'event+time':
            tf_dataset = tf_dataset.map(lambda x, y, t: (
                (tf.sparse.to_dense(x)[0:-1,:], tf.sparse.to_dense(y)[0:-1,:]), # remove last element from the sequence (EOS) event to present the input
                (tf.sparse.to_dense(x)[1:,:], tf.sparse.to_dense(y)[1:,:]) # remove first element from the sequence to present the target
            ))
            # one-hot encode output variable for classification branch
            tf_dataset = tf_dataset.map(lambda x, y: (x, (tf.one_hot(tf.squeeze(y[0]), n_states), y[1])))
    # dataset without log data information
    elif 'no_log' in context:
        if shuffle: tf_dataset = tf_dataset.shuffle(len(tf_dataset), reshuffle_each_iteration = True) # apply (perfect) shuffling
        if target == 'type':
            tf_dataset = tf_dataset.map(lambda x, y, z, c: (
                (tf.sparse.to_dense(c)[0:-1,:]), # remove last element from the sequence (EOS) event to present the input
                (tf.sparse.to_dense(z)[1:,:], ) # remove first element from the sequence to present the target
            ))
            # one-hot encode output variable for classification branch
            tf_dataset = tf_dataset.map(lambda x, y: (x, (tf.one_hot(tf.squeeze(y[0]), n_types), )))
        if target == 'event+time':
            tf_dataset = tf_dataset.map(lambda x, y, z, c: (
                (tf.sparse.to_dense(c)[0:-1,:]), # remove last element from the sequence (EOS) event to present the input
                (tf.sparse.to_dense(x)[1:,:], tf.sparse.to_dense(y)[1:,:]) # remove first element from the sequence to present the target
            ))
            # one-hot encode output variable for classification branch
            tf_dataset = tf_dataset.map(lambda x, y: (x, (tf.one_hot(tf.squeeze(y[0]), n_states), y[1])))
    # dataset with context
    else:
        if shuffle: tf_dataset = tf_dataset.shuffle(len(tf_dataset), reshuffle_each_iteration = True) # apply (perfect) shuffling
        if target == 'type':
            tf_dataset = tf_dataset.map(lambda x, y, z, c: (
                (tf.sparse.to_dense(x)[0:-1,:], tf.sparse.to_dense(y)[0:-1,:], tf.sparse.to_dense(c)[0:-1,:]), # remove last element from the sequence (EOS) event to present the input
                (tf.sparse.to_dense(z)[1:,:], ) # remove first element from the sequence to present the target
            ))
            # one-hot encode output variable for classification branch
            tf_dataset = tf_dataset.map(lambda x, y: (x, (tf.one_hot(tf.squeeze(y[0]), n_types), )))
        if target == 'event+time':
            tf_dataset = tf_dataset.map(lambda x, y, z, c: (
                (tf.sparse.to_dense(x)[0:-1,:], tf.sparse.to_dense(y)[0:-1,:], tf.sparse.to_dense(c)[0:-1,:]), # remove last element from the sequence (EOS) event to present the input
                (tf.sparse.to_dense(x)[1:,:], tf.sparse.to_dense(y)[1:,:]) # remove first element from the sequence to present the target
            ))
            # one-hot encode output variable for classification branch
            tf_dataset = tf_dataset.map(lambda x, y: (x, (tf.one_hot(tf.squeeze(y[0]), n_states), y[1])))

    # reshape volume if document features are used as context
    if 'doc_features' in context:
        if 'bert' not in context:
            if 'no_log' in context:
                tf_dataset = tf_dataset.map(lambda x, y: (tf.reshape(x, ((max_instance_length - 1), (max_page_n * 7), 7, 512)), y))
            else:
                tf_dataset = tf_dataset.map(lambda x, y: ((x[0], x[1], tf.reshape(x[2], ((max_instance_length - 1), (max_page_n * 7), 7, 512))), y))

    # configure batches
    tf_dataset = tf_dataset.batch(batch_size, drop_remainder = drop_remainder)

    return tf_dataset


# create neural network for predictive process monitoring with document context
def create_process_model(max_instance_length, n_states, n_types, max_page_n, context, target, img_feature_channels, doc_vector_size, lstm_layer_size, lstm_dropout):

    # define input sequences
    event_sequence_input = tf.keras.layers.Input(shape = (max_instance_length - 1,))
    event_sequence_embedding = tf.keras.layers.Embedding(n_states, 8)(event_sequence_input)
    time_sequence_input = tf.keras.layers.Input(shape = (max_instance_length - 1, 1))

    # plain rnn without context information
    if context == 'none':
        rnn_input = tf.keras.layers.Concatenate(axis = 2)([event_sequence_embedding, time_sequence_input])

    # rnn with number of pages as context information
    if context in ['doc_flags', 'doc_page_n']:
        # define input sequences
        context_sequence_input = tf.keras.layers.Input(shape = (max_instance_length - 1, 1))
        rnn_input = tf.keras.layers.Concatenate(axis = 2)([event_sequence_embedding, time_sequence_input, context_sequence_input])

    # rnn with document features as context information
    if 'doc_features' in context:
        # define context integration module as model
        feature_transformer = tf.keras.models.Sequential()

        # integrate bert embeddings of pages as context information
        if 'bert' in context:
            feature_transformer.add(tf.keras.layers.Input(shape = (max_page_n, 768)))
            feature_transformer.add(tf.keras.layers.LSTM(lstm_layer_size, return_sequences = True)) # activation is tanh per default
            feature_transformer.add(tf.keras.layers.LSTM(lstm_layer_size, return_sequences = False)) # activation is tanh per default
            feature_transformer.add(tf.keras.layers.Flatten())
            feature_transformer.add(tf.keras.layers.Dense(doc_vector_size, activation = 'relu')) # vector size per page

            # define input sequence for context
            context_sequence_input = tf.keras.layers.Input(shape = (max_instance_length - 1, max_page_n, 768))
        # integrate VGG features of pages as context information
        else:
            feature_transformer.add(tf.keras.layers.Input(shape = ((max_page_n * 7), 7, 512)))
            feature_transformer.add(tf.keras.layers.Conv2D(filters = img_feature_channels, kernel_size = (1, 1), activation = 'relu'))
            feature_transformer.add(tf.keras.layers.Conv2D(filters = img_feature_channels, kernel_size = (7, 7), strides = (7, 1), activation = 'relu'))
            feature_transformer.add(tf.keras.layers.Flatten())
            feature_transformer.add(tf.keras.layers.Dense(doc_vector_size, activation = 'relu')) # vector size per page

            # define input sequence for context
            context_sequence_input = tf.keras.layers.Input(shape = (max_instance_length - 1, (max_page_n * 7), 7, 512))

        # apply context integration module to each document in the event sequence
        distribute_over_documents = tf.keras.layers.TimeDistributed(feature_transformer)
        context_sequence = distribute_over_documents(context_sequence_input)

        # define input sequences without log data information
        if 'no_log' in context:
            rnn_input = context_sequence
        else:
            rnn_input = tf.keras.layers.Concatenate(axis = 2)([event_sequence_embedding, time_sequence_input, context_sequence])

    # define ppm model as LSTM
    shared_lstm = tf.keras.layers.LSTM(lstm_layer_size, return_sequences = True, dropout = lstm_dropout)(rnn_input) # activation is tanh per default

    # define complete input layer
    if 'no_log' in context:
        input_layers = []
    else:
        input_layers = [event_sequence_input, time_sequence_input]
    if context != 'none': input_layers.append(context_sequence_input)

    # define prediction branches and complete output layer
    output_layers = []
    # type prediction with one classification branch
    if target == 'type':

        type_cls_branch = tf.keras.layers.LSTM(lstm_layer_size, return_sequences = True, dropout = lstm_dropout)(shared_lstm) # activation is tanh per default
        type_cls_output = tf.keras.layers.Dense(n_types, activation = 'softmax', name = 'type_cls')(type_cls_branch)
        output_layers.append(type_cls_output)
    # event and time prediction with one classification and one regression branch
    if target == 'event+time':

        event_cls_branch = tf.keras.layers.LSTM(lstm_layer_size, return_sequences = True, dropout = lstm_dropout)(shared_lstm) # activation is tanh per default
        event_cls_output = tf.keras.layers.Dense(n_states, activation = 'softmax', name = 'event_cls')(event_cls_branch)
        output_layers.append(event_cls_output)

        time_reg_branch = tf.keras.layers.LSTM(lstm_layer_size, return_sequences = True, dropout = lstm_dropout)(shared_lstm) # activation is tanh per default
        time_reg_output = tf.keras.layers.Dense(1, activation = 'linear', name = 'time_reg')(time_reg_branch)
        output_layers.append(time_reg_output)

    # define keras model
    rnn_model = tf.keras.models.Model(input_layers, output_layers)

    return rnn_model

# calculate prediction metrics ignoring padding
def calculate_adjusted_metrics(y, y_hat, idx, target):

    if 'cls' in target:
        # calculate loss
        loss = np.mean(tf.keras.metrics.categorical_crossentropy(y, y_hat))
        # calculate mse on original sequences
        y = np.argmax(y, axis = 2)
        y_hat = np.argmax(y_hat, axis = 2)
        y = np.concatenate([i[0:j] for i, j in zip(y, idx)], axis = 0)
        y_hat = np.concatenate([i[0:j] for i, j in zip(y_hat, idx)], axis = 0)
        accuracy = np.sum(y == y_hat) / len(y)
        # return metrics
        return {(target + '_loss'): loss, (target + '_accuracy'): accuracy}

    elif 'reg' in target:
        # calculate loss
        loss = np.mean(tf.keras.metrics.mean_squared_error(y, y_hat))
        # calculate accuracy on original sequences
        y = np.concatenate([i[0:j] for i, j in zip(y, idx)], axis = 0)
        y_hat = np.concatenate([i[0:j] for i, j in zip(y_hat, idx)], axis = 0)
        mse = np.mean(tf.keras.metrics.mean_squared_error(y, y_hat))
        # return metrics
        return {(target + '_loss'): loss, (target + '_mse'): mse}

# run model evaluation on selected data fold & split and generate predictions
def evaluate_model(evaluation_object, tf_dataset, log_dataframe, target, data_split, return_predictions = False):

    # prepare indices for removing padded values
    log_dataframe = log_dataframe[log_dataframe.data_split == data_split].reset_index()
    idx = log_dataframe.groupby('instance_id')['instance_id'].count()

    # get actuals from tf dataset as numpy
    if target == 'type':
        type_cls_actuals = np.concatenate([i[1][0] for i in tf_dataset.as_numpy_iterator()], axis = 0)
    if target == 'event+time':
        event_cls_actuals = np.concatenate([i[1][0] for i in tf_dataset.as_numpy_iterator()], axis = 0)
        time_reg_actuals = np.concatenate([i[1][1] for i in tf_dataset.as_numpy_iterator()], axis = 0)

    # get model predictions
    if isinstance(evaluation_object, tf.keras.models.Model):
        prediction = evaluation_object.predict(tf_dataset, verbose = 0)
    else:
        prediction = evaluation_object

    if target == 'type':
        # calculate classification metrics for type classifiation
        metrics = calculate_adjusted_metrics(type_cls_actuals, prediction, idx, target = 'type_cls')
        # calculate total loss
        total_loss = 1 * metrics['type_cls_loss']

    if target == 'event+time':
        # calculate classification metrics for event classification
        metrics = calculate_adjusted_metrics(event_cls_actuals, prediction[0], idx, target = 'event_cls')
        # calculate regression metrics for time regression
        metrics.update(calculate_adjusted_metrics(time_reg_actuals, prediction[1], idx, target = 'time_reg'))
        # calculate total loss
        total_loss = 1 * metrics['event_cls_loss'] + 1 * metrics['time_reg_loss']

    # return individual predictions for unpadded sequences
    if return_predictions:
        if target == 'type':
            prediction_df = pd.DataFrame({
                'instance_id': log_dataframe.instance_id.values,
                'type_cls_actuals': np.argmax(np.concatenate([i[0:j] for i, j in zip(type_cls_actuals, idx)], axis = 0), axis = 1),
                'type_cls_prediction': np.argmax(np.concatenate([i[0:j] for i, j in zip(prediction, idx)], axis = 0), axis = 1)
            })
            class_probabilities = pd.DataFrame(np.concatenate([i[0:j] for i, j in zip(prediction, idx)], axis = 0))
            prediction_df = pd.concat([prediction_df, class_probabilities], axis = 1)
        if target == 'event+time':
            prediction_df = pd.DataFrame({
                'instance_id': log_dataframe.instance_id.values,
                'event_cls_actuals': np.argmax(np.concatenate([i[0:j] for i, j in zip(event_cls_actuals, idx)], axis = 0), axis = 1),
                'event_cls_prediction': np.argmax(np.concatenate([i[0:j] for i, j in zip(prediction[0], idx)], axis = 0), axis = 1),
                'time_reg_actuals': np.concatenate([i[0:j] for i, j in zip(time_reg_actuals, idx)], axis = 0).squeeze(),
                'time_reg_prediction': np.concatenate([i[0:j] for i, j in zip(prediction[1], idx)], axis = 0).squeeze()
            })
            class_probabilities = pd.DataFrame(np.concatenate([i[0:j] for i, j in zip(prediction[0], idx)], axis = 0))
            prediction_df = pd.concat([prediction_df, class_probabilities], axis = 1)

        return ({'data_split': data_split, 'total_loss': total_loss, **metrics}, prediction_df)
    else:
        return {'data_split': data_split, 'total_loss': total_loss, **metrics}

# callback to record actual predictions after each epoch
class PredictionHistoryCallback(tf.keras.callbacks.Callback):

    def __init__(self, val_dataset):
        self.val_dataset = val_dataset
        self.predictions_history = []

        super().__init__()

    def on_epoch_end(self, epoch, logs = None):
        # generate and store predictions on current epoch
        predictions = self.model.predict(self.val_dataset)
        self.predictions_history.append(predictions)

# train parameter configuration and evaluate on validation set for selected fold
def run_training_configuration(rnn_params, dataset_params, training_params, tf_dataset):

    # configure datasets
    train_data = configure_tf_dataset(tf_dataset['train_set'], drop_remainder = True, shuffle = True, **dataset_params)
    val_data = configure_tf_dataset(tf_dataset['val_set'], drop_remainder = False, shuffle = False, **dataset_params)

    # create model
    rnn = create_process_model(**rnn_params)

    # compile model
    if rnn_params['target'] == 'type':
        rnn.compile(
            optimizer = tf.keras.optimizers.RMSprop(learning_rate = training_params['learning_rate']),
            loss = {'type_cls': 'categorical_crossentropy'},
            loss_weights = {'type_cls': 1}
        )
    if rnn_params['target'] == 'event+time':
        rnn.compile(
            optimizer = tf.keras.optimizers.RMSprop(learning_rate = training_params['learning_rate']),
            loss = {'event_cls': 'categorical_crossentropy', 'time_reg': 'mse'},
            loss_weights = {'event_cls': 1, 'time_reg': 1}
        )

    # set up callback to record prediction history
    callbacks = []
    if training_params['record_prediction_history']:
        prediction_history_recorder = PredictionHistoryCallback(val_data)
        callbacks.append(prediction_history_recorder)

    if training_params['early_stopping']:
        early_stopper = tf.keras.callbacks.EarlyStopping(
            monitor = 'val_loss',
            patience = training_params['es_patience'],
            verbose = 1,
            restore_best_weights=False,
        )
        callbacks.append(early_stopper)

    # set up callback to save best model
    os.makedirs('tmp/', exist_ok = True)
    # define path for temporary model checkpoint
    checkpoint_path = 'tmp/model_checkpoint.hdf5'
    checkpointer = tf.keras.callbacks.ModelCheckpoint(
        filepath = checkpoint_path,
        monitor = 'val_loss',
        save_weights_only = True,
        save_best_only = True,
        verbose = training_params['verbosity']
    )
    callbacks = callbacks + [checkpointer]

    # train model
    with tf.device(training_params['device']):
        history = rnn.fit(
            train_data,
            epochs = training_params['epochs'],
            validation_data = val_data,
            verbose = training_params['verbosity'],
            callbacks = callbacks
        )

    # load weights from best checkpoint
    rnn.load_weights(checkpointer.filepath)

    # configure training dataset without shuffling
    train_data = configure_tf_dataset(tf_dataset['train_set'], drop_remainder = True, shuffle = False, **dataset_params) # remove shuffling

    # evaluate model on training set
    training = evaluate_model(
        rnn,
        train_data,
        dataset_params['log_dataframe'],
        target = rnn_params['target'],
        data_split = 'train_set'
    )

    # evaluate model on validation set
    validation = evaluate_model(
        rnn,
        val_data,
        dataset_params['log_dataframe'],
        target = rnn_params['target'],
        data_split = 'val_set'
    )

    # create evaluation dataframe
    evaluation = pd.DataFrame([training, validation])
    evaluation['context'] = rnn_params['context']
    evaluation['target'] = rnn_params['target']
    evaluation['best_epoch'] = np.argmin(history.history['val_loss']) + 1

    # create training history dataframe
    training_history = pd.DataFrame(history.history)
    training_history['context'] = rnn_params['context']
    training_history['target'] = rnn_params['target']
    training_history['best_epoch'] = np.argmin(history.history['val_loss']) + 1

    # create output dictionary
    result_dict = {
        'evaluation': evaluation,
        'training_history': training_history,
        'model': rnn
    }
    if training_params['record_prediction_history']:
        result_dict['predictions_history'] = prediction_history_recorder.predictions_history

    return result_dict

# function to load best model together with the test set for selected fold
def load_best_model_and_test_data(target, context, fold, log_df, folds_df, eval_df_path, model_dir, max_instance_length, max_page_n, n_states, n_types):

    # load model evaluation & select best model for requested fold
    evaluation = pd.read_csv(eval_df_path)
    best_config = evaluation[(evaluation.context == context) & (evaluation.fold == fold)].iloc[0]

    # select fold
    selected_fold = folds_df[['instance_id', f'data_split_{fold}']].rename(columns = {f'data_split_{fold}': 'data_split'})
    selected_fold = selected_fold.merge(log_df, on = 'instance_id', how = 'inner')

    # set-up and load tf_dataset
    dataset_params = {
        'log_dataframe': selected_fold,
        'context': context,
        'target': target,
        'max_instance_length': max_instance_length,
        'max_page_n': max_page_n,
        'n_states': n_states,
        'n_types': n_types,
        'batch_size': 1
    }

    if 'no_log' in context:
        dataset_context = context[:-7]
    else:
        dataset_context = context

    test_data = tf.data.Dataset.load(f'tf_datasets/{dataset_context}/fold_{fold}/test_set')
    test_data = configure_tf_dataset(test_data, drop_remainder = False, shuffle = False, **dataset_params)

    # load model
    model_dir = f"results/trained_models/{target.replace('+', '_')}"
    rnn = tf.keras.models.load_model(f"{model_dir}/{context}_fold_{fold}_grid_row_{best_config.grid_row}.h5")

    # select test set from log data
    log_df_test_set = selected_fold[selected_fold.data_split == 'test_set']

    return [rnn, test_data, log_df_test_set]
