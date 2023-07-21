# Author: Sergej Levich
# Journal article: Sergej Levich et al., Decision Support Systems, https://doi.org/10.1016/j.dss.2023.114043

from functions import *
from sklearn.metrics import pairwise_distances
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

# function to extract document representation from models selected models
def extract_document_representations(target, context, fold, output_path):

    # set up paths for evaluation file and models
    eval_df_path = 'results/evaluation/test_set/type/evaluation.csv'
    model_dir = f"results/trained_models/{target.replace('+', '_')}"

    # load model and test set data
    rnn, test_data, log_df_test_set = load_best_model_and_test_data(
        target = target,
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

    # extract document encoder that yields document embeddings
    doc_encoder = rnn.layers[4].layer # getting sequential model from the time distributed layer

    # get pretrained features from the VGG model and embeddings from the document encoder for each document and each page
    doc_df = log_df_test_set[['instance_id', 'state_name', 'type_name', 'file_name', 'n_pages']].copy()
    doc_df = doc_df.reset_index(drop = True)
    doc_df['pretrained_doc_features'] = None
    doc_df['learned_doc_embeddings'] = None
    instance_ids = doc_df.instance_id.unique()

    for i, sample in tqdm(enumerate(test_data)):
        doc_input = sample[0][2][0]
        encoded_sample_docs = doc_encoder(doc_input).numpy()
        instance_df = doc_df[doc_df.instance_id == instance_ids[i]]

        for j, idx in enumerate(instance_df.index):
            n_pages = doc_df.loc[idx, 'n_pages']
            if n_pages != 0:
                n_pages = int(n_pages)
                doc_df.loc[idx, 'pretrained_doc_features'] = [doc_input[j].numpy()]
                doc_df.loc[idx, 'learned_doc_embeddings'] = [encoded_sample_docs[j,:, np.newaxis]]


    doc_df = doc_df[doc_df.file_name != ''].copy()
    doc_df['doc_id'] = range(len(doc_df))
    doc_df['page_nr'] = [(np.arange(i) + 1) for i in doc_df['n_pages']]
    doc_df = doc_df.reset_index(drop = True)
    doc_df.pretrained_doc_features = list(np.concatenate(doc_df.pretrained_doc_features))
    doc_df.learned_doc_embeddings = list(np.concatenate(doc_df.learned_doc_embeddings))

    # calculate similarity matrices
    pretrained_doc_similarities = 1 - pairwise_distances([i.ravel() for i in doc_df['pretrained_doc_features']], metric = 'cosine')
    learned_doc_similarities = 1 - pairwise_distances([i.ravel() for i in doc_df['learned_doc_embeddings']], metric = 'cosine')

    # prepare output folder
    output_path = os.path.join(output_path, target, context)
    os.makedirs(output_path, exist_ok = True)

    # save cosine similarities as dataframe
    doc_similarity_df = pd.DataFrame({
        'doc_id_a': np.triu_indices_from(pretrained_doc_similarities, k = 1)[0],
        'doc_id_b': np.triu_indices_from(pretrained_doc_similarities, k = 1)[1],
        'pretrained_similarity': pretrained_doc_similarities[np.triu_indices_from(pretrained_doc_similarities, k = 1)],
        'learned_similarity': learned_doc_similarities[np.triu_indices_from(learned_doc_similarities, k = 1)]
    })
    doc_similarity_df.shape
    doc_similarity_df.to_csv(os.path.join(output_path, 'doc_similarity_df.csv'), index = False)

    # save document information
    doc_df[['instance_id', 'state_name', 'type_name', 'file_name', 'n_pages', 'doc_id']].to_csv(os.path.join(output_path, 'doc_df.csv'), index = False)

    # save learned document embedding vectors as csv
    pd.DataFrame(np.array([i for i in doc_df['learned_doc_embeddings']]).squeeze()).to_csv(os.path.join(output_path, 'learned_doc_embeddings.csv'), index = False)

    print(f'document representations extracted for context {context} and target {target} based on test set from fold {fold}')


# define models for requested document representations
model_config = pd.DataFrame({
    'context': [
        'doc_features_imagenet',
        'doc_features_rvl',
        'doc_features_bert_layoutxlm',
        'doc_features_bert_german'
    ],
    'target': [
        'type',
        'type',
        'type',
        'type'
    ]
})

# run extraction
for i, row in model_config.iterrows():

    # extract document representations from model for selected target and context
    extract_document_representations(
        target = row.target,
        context = row.context,
        fold = 0,
        output_path = 'results/embeddings'
    )
