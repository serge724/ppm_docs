# Author: Sergej Levich
# Journal article: Sergej Levich et al., Decision Support Systems, https://doi.org/10.1016/j.dss.2023.114043

from functions import *

# load log data & fold splits
log_df = pd.read_csv('log_data/process_log.csv')
log_df = log_df.fillna({'file_name': '', 'n_pages': 0, 'time_to_next_state': 0})
folds_df = pd.read_csv('log_data/folds_and_splits.csv')

# define process parameters
n_states = log_df.state.value_counts().size + 1 # adding end state as class
n_types = log_df.type.value_counts().size
max_instance_length = log_df.instance_id.value_counts().max() + 1 # adding end state to sequence length
max_page_n = 10

# function to evaluate best performing configuration on test set for each fold
def evaluate_on_test_set(target):

    # set up paths for evaluation files and models
    eval_dir = f"results/evaluation/val_set/{target.replace('+', '_')}"
    model_dir = f"results/trained_models/{target.replace('+', '_')}"

    # load validation set results
    eval_files = os.listdir(eval_dir)
    evaluation = []
    for i in eval_files:
        evaluation.append(pd.read_csv(f"{eval_dir}/{i}"))

    # find best model configuration per fold
    evaluation = pd.concat(evaluation)
    evaluation = evaluation[evaluation.data_split == 'val_set'].reset_index(drop = True)
    best_configs = evaluation.groupby(['context', 'fold']).apply(lambda x: x[x.total_loss == x.total_loss.min()])
    best_configs = best_configs.reset_index(drop = True)

    # run models on test set
    test_set_evaluation = []
    test_set_predictions = []
    for i, config in best_configs.iterrows():

        print(f'evaluating {config.context} grid row {config.grid_row} on fold {config.fold}')

        # select fold
        selected_fold = folds_df[['instance_id', f'data_split_{config.fold}']].rename(columns = {f'data_split_{config.fold}': 'data_split'})
        selected_fold = selected_fold.merge(log_df, on = 'instance_id', how = 'inner')

        # set-up dataset
        dataset_params = {
            'log_dataframe': selected_fold,
            'context': config.context,
            'target': config.target,
            'max_instance_length': max_instance_length,
            'max_page_n': max_page_n,
            'n_states': n_states,
            'n_types': n_types,
            'batch_size': 16
        }

        if 'no_log' in config.context:
            dataset_context = config.context[:-7]
        else:
            dataset_context = config.context

        test_data = tf.data.Dataset.load(f'tf_datasets/{dataset_context}/fold_{config.fold}/test_set')
        test_data = configure_tf_dataset(test_data, drop_remainder = False, shuffle = False, **dataset_params)

        # load model
        rnn = tf.keras.models.load_model(f"{model_dir}/{config.context}_fold_{config.fold}_grid_row_{config.grid_row}.h5")

        # evaluate
        evaluation = evaluate_model(
            rnn,
            test_data,
            selected_fold,
            target = target,
            data_split = 'test_set',
            return_predictions = True
        )

        # add model info to evaluation results
        testing = pd.DataFrame([evaluation[0]])
        testing['context'] = config.context
        testing['target'] = config.target
        testing['grid_row'] = config.grid_row
        testing['fold'] = config.fold
        testing['learning_rate'] = config.learning_rate
        testing['layer_size'] = config.layer_size
        test_set_evaluation.append(testing)

        # add model info to predictions
        predictions = evaluation[1]
        predictions['context'] = config.context
        predictions['target'] = config.target
        predictions['grid_row'] = config.grid_row
        predictions['fold'] = config.fold
        predictions['learning_rate'] = config.learning_rate
        predictions['layer_size'] = config.layer_size
        test_set_predictions.append(predictions)

    # create output directory
    testing_dir = f"results/evaluation/test_set/{target.replace('+', '_')}"
    os.makedirs(testing_dir, exist_ok = True)

    # save results
    test_set_evaluation = pd.concat(test_set_evaluation, ignore_index = True)
    test_set_evaluation.to_csv(f"{testing_dir}/evaluation.csv", index = False)
    test_set_predictions = pd.concat(test_set_predictions, ignore_index = True)
    test_set_predictions.to_csv(f"{testing_dir}/predictions.csv", index = False)

    print('best models for target %s evaluated on test set'%(target))


# evaluate best models on test set for type target
evaluate_on_test_set(target = 'type')

# evaluate best models on test set for event & time targets
evaluate_on_test_set(target = 'event+time')
