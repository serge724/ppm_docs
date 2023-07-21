# Author: Sergej Levich
# Journal article: Sergej Levich et al., Decision Support Systems, https://doi.org/10.1016/j.dss.2023.114043

import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tqdm import tqdm

# load log data & fold splits
log_df = pd.read_csv('log_data/process_log.csv')
log_df = log_df.fillna({'file_name': '', 'n_pages': 0, 'time_to_next_state': 0})
folds_df = pd.read_csv('log_data/folds_and_splits.csv')

# define process parameters
n_states = log_df.state.value_counts().size + 1 # adding end state as class
max_instance_length = log_df.instance_id.value_counts().max() + 1 # adding end state to sequence length
max_page_n = 10

# function to generate numpy arrays for each process instance
def get_arrays_from_instance(instance_df, context, path_to_extracted_features, n_states, max_instance_length, max_page_n):

    arrays = {}

    # create sequences of process features
    state_sequence = np.array(instance_df.state.values)
    state_sequence = np.pad(state_sequence, (0, max_instance_length - len(state_sequence)), constant_values = (n_states - 1)) # add EOS event & padding
    arrays['states'] = state_sequence

    time_sequence = np.array(instance_df.log_time_since_last_event.values, dtype = 'float32')
    time_sequence = np.pad(time_sequence, (0, max_instance_length - len(time_sequence)), constant_values = 0) # add time until EOS event (=0) & padding
    arrays['times'] = time_sequence

    type_sequence = np.array(instance_df.type.values)
    type_sequence = np.pad(type_sequence, (0, max_instance_length - len(type_sequence)), mode = 'edge') # add EOS event & padding
    arrays['types'] = type_sequence

    # create context sequence
    if context != 'none':
        context_sequence = []
        for j, file in enumerate(instance_df.file_name):

            # if document is associated to state, apply CNN model
            if not file == '':
                # load context features
                if 'doc_features' in context:
                    feature_paths = [os.path.join(path_to_extracted_features, j) for j in os.listdir(path_to_extracted_features) if j[:5] == file[:5]]
                    feature_paths.sort()
                    event_context = [np.load(k) for k in feature_paths]
                    event_context = np.concatenate(event_context, axis = 0, dtype = 'float32')
                    # pad context to match maximum number of pages
                    if 'bert' in context:
                        event_context = np.pad(event_context, ((0, max_page_n - event_context.shape[0]), (0, 0)))
                    else:
                        event_context = np.pad(event_context, ((0, max_page_n - event_context.shape[0]), (0, 0), (0, 0), (0, 0)))
                # or assign flag
                if context == 'doc_flags':
                    event_context = 1
                # or read number of pages from log
                if context == 'doc_page_n':
                    event_context = instance_df.n_pages.values[j] / max_page_n # scale number of pages
            # if no document is associated to state, create array of zeros
            else:
                if 'doc_features' in context:
                    if 'bert' in context:
                        event_context = np.zeros((max_page_n, 768), dtype = 'float32')
                    else:
                        event_context = np.zeros((max_page_n, 7, 7, 512), dtype = 'float32')
                if context == 'doc_flags':
                    event_context = 0
                if context == 'doc_page_n':
                    event_context = 0

            context_sequence.append(event_context)

        context_sequence = np.array(context_sequence, dtype = 'float32')
        if 'doc_features' in context:
            if 'bert' in context:
                context_sequence = np.pad(context_sequence, ((0, max_instance_length - context_sequence.shape[0]), (0, 0), (0, 0)))
            else:
                context_sequence = np.pad(context_sequence, ((0, max_instance_length - context_sequence.shape[0]), (0, 0), (0, 0), (0, 0), (0, 0)))
        if context == 'doc_flags':
            context_sequence = np.pad(context_sequence[:,np.newaxis], ((0, max_instance_length - context_sequence.shape[0]), (0, 0)))
        if context == 'doc_page_n':
            context_sequence = np.pad(context_sequence[:,np.newaxis], ((0, max_instance_length - context_sequence.shape[0]), (0, 0)))

        arrays['context'] = context_sequence

    return arrays

# function to create a tensorflow dataset from a dataframe with log data
def create_tf_dataset(log_df, context, path_to_extracted_features, n_states, max_instance_length, max_page_n):

    # create lists for process instances
    sequences = {}
    for i in ['states', 'times', 'types']:
        sequences[i] = []
    if context != 'none':
        sequences['context'] = []

    for id in tqdm(log_df.instance_id.unique()):

        # select instance
        instance_df = log_df[log_df.instance_id == id]

        # generate arrays
        instance_arrays = get_arrays_from_instance(
            instance_df,
            context = context,
            path_to_extracted_features = path_to_extracted_features,
            n_states = n_states,
            max_instance_length = max_instance_length,
            max_page_n = max_page_n
        )

        # append to ordered lists of process instances
        for i, j in instance_arrays.items():
            # transform context sequence to sparse tensor to reduce memory requirements
            array = tf.sparse.from_dense(j[np.newaxis,:]) if i == 'context' else j[:,np.newaxis]
            sequences[i].append(array)

    # transform other sequences to sparse tensors to reduce memory requirements
    for i, j in sequences.items():
        sequences[i] = tf.sparse.concat(0, j) if i == 'context' else tf.sparse.from_dense(j)

    # create tensorflow dataset
    return tf.data.Dataset.from_tensor_slices(tuple(sequences.values()))


# define requested datasets
dataset_config = pd.DataFrame({
    'context': [
        'none',
        'doc_flags',
        'doc_page_n',
        'doc_features_rvl',
        'doc_features_imagenet',
        'doc_features_bert_german',
        'doc_features_bert_layoutxlm'
    ],
    'path': [
        None,
        None,
        None,
        'processed_data/features/vgg_rvl',
        'processed_data/features/vgg_imagenet',
        'processed_data/features/bert_german',
        'processed_data/features/bert_layoutxlm'
    ]
})

# create requested datasets
n_folds = folds_df.filter(regex = ('data_split')).shape[1]
# for each context
for i, row in dataset_config.iterrows():
    print(f'running context: {row.context}')

    # for each fold
    for fold in range(n_folds):
        print(f'running fold: {fold}')
        selected_fold = folds_df[['instance_id', f'data_split_{fold}']].rename(columns = {f'data_split_{fold}': 'data_split'})
        selected_fold = selected_fold.merge(log_df, on = 'instance_id', how = 'inner')

        # for each split
        for split in ('train_set', 'val_set', 'test_set'):
            print(f'running split: {split}')

            # skip if dataset exists
            if os.path.exists(f'tf_datasets/{row.context}/fold_{fold}/{split}'):
                continue

            # create dataset
            selected_split = selected_fold[selected_fold.data_split == split].drop(columns = ['data_split'])
            with tf.device('CPU:0'):
                tf_dataset = create_tf_dataset(
                    log_df = selected_split,
                    context = row.context,
                    path_to_extracted_features = row.path,
                    n_states = n_states,
                    max_instance_length = max_instance_length,
                    max_page_n = max_page_n
                )

            tf_dataset.save(f'tf_datasets/{row.context}/fold_{fold}/{split}')

print(f'tensorflow datasets for {len(dataset_config)} contexts created')
