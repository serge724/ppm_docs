# Author: Sergej Levich
# Journal article: Sergej Levich et al., Decision Support Systems, https://doi.org/10.1016/j.dss.2023.114043

from functions import *
import gc
import shap
from tqdm import tqdm

# load log data & fold splits
log_df = pd.read_csv('log_data/process_log.csv')
log_df = log_df.fillna({'file_name': '', 'n_pages': 0, 'time_to_next_state': 0})
folds_df = pd.read_csv('log_data/folds_and_splits.csv')

# define process parameters
n_states = log_df.state.value_counts().size + 1 # adding end state as class
n_types = log_df.type.value_counts().size
max_instance_length = log_df.instance_id.value_counts().max() + 1 # adding end state to sequence length
max_page_n = 10

# function to replace context information with background with masks selected by shap.KernelExplainer
def mask_instance(masks, sample_input, background):
    event_input, time_input, context_input = sample_input
    event = np.zeros((masks.shape[0], event_input.shape[1], event_input.shape[2]))
    time = np.zeros((masks.shape[0], time_input.shape[1], time_input.shape[2]))
    mask_shape = masks.shape[0] + context_input.shape[1:]
    masked_context = np.zeros(mask_shape)
    for i in range(masks.shape[0]):
        event[i,:,:] = event_input
        time[i,:,:] = time_input
        masked_context[i,:,:] = context_input
        for j in range(masks.shape[1]):
            if masks[i, j] == 0:
                masked_context[i, j, :] = background

    return [event, time, masked_context]

# predictor that returns predictions based on masked input
def masked_predictor(model, event_nr, masks, sample_input, background):
    full_sequnce_prediction = model.predict(mask_instance(masks, sample_input, background), batch_size = 16, verbose = 0)
    return full_sequnce_prediction[:, event_nr, :]

# function to calculate shap values for each sample of the test set for type target
def calculate_type_shap_values(context, fold, output_path):

    # set up paths for evaluation file and models
    eval_df_path = 'results/evaluation/test_set/type/evaluation.csv'
    model_dir = 'results/trained_models/type'

    # load model and test set data
    rnn, test_data, log_df_test_set = load_best_model_and_test_data(
        target = 'type',
        context = context,
        fold = fold,
        log_df = log_df,
        folds_df = folds_df,
        eval_df_path = eval_df_path,
        model_dir = model_dir,
        max_instance_length = max_instance_length,
        max_page_n = max_page_n,
        n_states = n_states,
        n_types = n_types
    )

    # find context input tensor that correspons to the background distribution (no document)
    bg_index = np.where(log_df_test_set.file_name == '')[0][0]
    input = [input for (input, output) in test_data.take(1)][0]
    background = input[2][0][bg_index].numpy()

    # calculate shap values using KernelExplainer
    shap_results = []
    for id, (input, output) in tqdm(zip(log_df_test_set.instance_id.unique(), test_data), total = len(test_data)):
        instance_df = log_df_test_set[log_df_test_set.instance_id == id]
        explainer = shap.KernelExplainer(
            lambda x: masked_predictor(rnn, len(instance_df)-1, x, input, background),
            np.zeros((1, len(instance_df)))
        )
        shap_values = explainer.shap_values(np.ones((1, len(instance_df))), nsamples = 500, silent = True)
        shap_values = pd.DataFrame(np.array(shap_values).squeeze())
        shap_values = shap_values.transpose()[0:len(instance_df)]
        shap_results.append(shap_values)
        gc.collect()

    # prepare output dataframe
    shap_df = pd.concat(
        [log_df_test_set.reset_index(drop = True), pd.concat(shap_results, axis = 0).reset_index(drop = True)],
        axis = 1
    )
    shap_df['context'] = context
    shap_df['target'] = 'type'
    shap_df['fold'] = 0

    # save results
    os.makedirs(output_path, exist_ok = True)
    shap_df.to_csv(os.path.join(output_path, 'shap_df.csv'), index = False)

    print(f'shap values for context {context} and target type calculated on test set from fold {fold}')


# calculate shap values for the best image-based model
calculate_type_shap_values(
    context = 'doc_features_rvl',
    fold = 0,
    output_path = 'results/shap/doc_features_rvl'
)

# calculate shap values for the best bert-based model
calculate_type_shap_values(
    context = 'doc_features_bert_german',
    fold = 0,
    output_path = 'results/shap/doc_features_bert_german'
)
