# Author: Sergej Levich
# Journal article: Sergej Levich et al., Decision Support Systems, https://doi.org/10.1016/j.dss.2023.114043

from functions import *
from itertools import product
from datetime import datetime

# load log data & fold splits
log_df = pd.read_csv('log_data/process_log.csv')
log_df = log_df.fillna({'file_name': '', 'n_pages': 0, 'time_to_next_state': 0})
folds_df = pd.read_csv('log_data/folds_and_splits.csv')

# define process parameters
n_states = log_df.state.value_counts().size + 1 # adding end state as class
n_types = log_df.type.value_counts().size
max_instance_length = log_df.instance_id.value_counts().max() + 1 # adding end state to sequence length
max_page_n = 10

# function to train models on a defined grid of model configurations
def search_grid(grid, log_df, folds_df):

    n_folds = folds_df.filter(regex = ('data_split')).shape[1]

    # for each model configuration
    for i, config in grid.iterrows():
        print(f'running grid row: {i}')

        # prepare folders for saving results
        eval_dir = f"results/evaluation/val_set/{config['target'].replace('+', '_')}"
        os.makedirs(eval_dir, exist_ok = True)
        history_dir = f"results/training_history/{config['target'].replace('+', '_')}"
        os.makedirs(history_dir, exist_ok = True)
        model_dir = f"results/trained_models/{config['target'].replace('+', '_')}"
        os.makedirs(model_dir, exist_ok = True)

        # for each fold
        for fold in range(n_folds):
            print(f'running fold: {fold}')

            # clear backend
            tf.keras.backend.clear_session()

            # skip training configuration if results are already available
            if os.path.exists(f"{model_dir}/{config['context']}_fold_{fold}_grid_row_{i}.h5"):
                continue

            # get data split of selected fold
            selected_fold = folds_df[['instance_id', f'data_split_{fold}']].rename(columns = {f'data_split_{fold}': 'data_split'})
            selected_fold = selected_fold.merge(log_df, on = 'instance_id', how = 'inner')

            # set up modeling parameters
            dataset_params = {
                'log_dataframe': selected_fold,
                'context': config['context'],
                'target': config['target'],
                'max_instance_length': max_instance_length,
                'max_page_n': max_page_n,
                'n_states': n_states,
                'n_types': n_types,
                'batch_size': 16
            }
            rnn_params = {
                'max_instance_length': max_instance_length,
                'n_states': n_states,
                'n_types': n_types,
                'max_page_n': max_page_n,
                'context': config['context'],
                'target': config['target'],
                'img_feature_channels': config['layer_size'],
                'doc_vector_size': config['layer_size'],
                'lstm_layer_size': config['layer_size'],
                'lstm_dropout': 0
            }
            training_params = {
                'epochs': 100,
                'record_prediction_history': False,
                'early_stopping': True,
                'es_patience': 5,
                'learning_rate': config['learning_rate'],
                'verbosity': 0,
                'device': 'GPU:0'
            }

            # load datasets for requested context and fold
            tf_dataset = {}
            for split in ('train_set', 'val_set', 'test_set'):
                if 'no_log' in dataset_params["context"]:
                    dataset_context = dataset_params["context"][:-7]
                else:
                    dataset_context = dataset_params["context"]
                tf_dataset[split] = tf.data.Dataset.load(f'tf_datasets/{dataset_context}/fold_{fold}/{split}')

            # train model with requested parameters
            run_results = run_training_configuration(rnn_params, dataset_params, training_params, tf_dataset)

            # add info on run
            run_results['evaluation']['grid_row'] = i
            run_results['evaluation']['fold'] = fold
            run_results['training_history']['grid_row'] = i
            run_results['training_history']['fold'] = fold
            for item, value in config.items():
                run_results['evaluation'][item] = value
                run_results['training_history'][item] = value

            # save results
            run_results['evaluation'].to_csv(f"{eval_dir}/{config['context']}_fold_{fold}_grid_row_{i}.csv", index = False)
            run_results['training_history'].to_csv(f"{history_dir}/{config['context']}_fold_{fold}_grid_row_{i}.csv", index = False)
            run_results['model'].save(f"{model_dir}/{config['context']}_fold_{fold}_grid_row_{i}.h5")

# define model configuration grid for type target
tuning_parameters = {
    'context': [
        'none',
        'doc_flags',
        'doc_page_n',
        'doc_features_rvl',
        'doc_features_imagenet',
        'doc_features_bert_german',
        'doc_features_bert_layoutxlm',
        'doc_features_bert_german_no_log'
    ],
    'target': ['type'],
    'learning_rate': [0.1, 0.01, 0.001],
    'layer_size': [16, 32, 64]
}
grid = [i for i in product(*tuning_parameters.values())]
grid = pd.DataFrame(grid, columns = tuning_parameters.keys())

# search grid
print('Searching following grid:')
print(grid)
search_grid(grid, log_df, folds_df)

# define model configuration grid for event & time targets
tuning_parameters = {
    'context': [
        'none',
        'doc_flags',
        'doc_page_n',
        'doc_features_rvl',
        'doc_features_imagenet',
        'doc_features_bert_german',
        'doc_features_bert_layoutxlm',
        'doc_features_bert_german_no_log'
    ],
    'target': ['event+time'],
    'learning_rate': [0.1, 0.01, 0.001],
    'layer_size': [16, 32, 64]
}
grid = [i for i in product(*tuning_parameters.values())]
grid = pd.DataFrame(grid, columns = tuning_parameters.keys())

# search grid
print('Searching following grid:')
print(grid)
search_grid(grid, log_df, folds_df)
