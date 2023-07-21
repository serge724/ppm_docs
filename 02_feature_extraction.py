# Author: Sergej Levich
# Journal article: Sergej Levich et al., Decision Support Systems, https://doi.org/10.1016/j.dss.2023.114043

import os
import json
import numpy as np
from PIL import ImageOps
from tqdm import tqdm
from doc2data.pdf import PDFCollection
from doc2data.utils import load_image

# function to create feature extractors
def create_feature_extractor(extractor):

    # define VGG-16 feature extractor pretrained on ImageNet
    if extractor == 'vgg_imagenet':
        import tensorflow as tf

        vgg_model = tf.keras.applications.VGG16(include_top = False, weights = 'imagenet', input_shape = (224, 224, 3))

        def feature_extractor(image):

            image = np.array(image)
            image = tf.keras.applications.vgg16.preprocess_input(image[np.newaxis,])
            features = vgg_model(image).numpy()

            return features

    # define VGG-16 feature extractor pretrained on RVL-CDIP
    # paper: https://arxiv.org/abs/1704.03557
    # repo: https://github.com/tuannamnguyen93/DFKI_test_PhD
    elif extractor == 'vgg_rvl':
        import tensorflow as tf

        # model definition copied from: https://github.com/tuannamnguyen93/DFKI_test_PhD/blob/master/models/all_model.py
        def define_vgg(n_classes,IMG_SIZE,dataset,use_imagenet=True):
            # load pre-trained model graph, don't add final layer
            model = tf.keras.applications.VGG16(include_top=False, input_shape=(IMG_SIZE, IMG_SIZE, 3),
                                                  weights='imagenet' if use_imagenet else None)
            # add global pooling just like in InceptionV3
            new_output = tf.keras.layers.Flatten(name='flatten')(model.output)
            new_output =   tf.keras.layers.Dense(4096, activation='relu', name='fc1')(new_output)

            new_output = tf.keras.layers.Dense(4096, activation='relu', name='fc2')(new_output)

            # add new dense layer for our labels
            new_output = tf.keras.layers.Dense(n_classes, activation='softmax',name='fc_'+dataset)(new_output)
            model = tf.keras.models.Model(model.inputs, new_output)
            return model

        vgg_model = define_vgg(n_classes = 16, IMG_SIZE = 224, dataset = 'RVL', use_imagenet = True)

        # load checkpoint wights from repository
        vgg_model.load_weights('checkpoint_RVL_CDIP_vgg_pretrained/weights-05-0.89.hdf5')

        # remove top layers
        vgg_model = tf.keras.models.Sequential(vgg_model.layers[0:19])

        def feature_extractor(image):

            # convert to grayscale
            image = np.array(ImageOps.grayscale(image))[:, :, np.newaxis]
            image = np.repeat(image, 3, axis = 2) / 255
            features = vgg_model(image[np.newaxis,]).numpy()

            return features

    # define BERT feature extractor pretrained on a collection of German text data
    # https://huggingface.co/bert-base-german-cased
    elif extractor == 'bert_german':
        from transformers import BertTokenizer, BertModel

        bert_tokenizer = BertTokenizer.from_pretrained("bert-base-german-cased")
        bert_model = BertModel.from_pretrained("bert-base-german-cased")

        def feature_extractor(token_dict):

            if token_dict != []:
                token_sequence = ' '.join([i['text'] for i in token_dict])
            else:
                token_sequence = ''

            inputs = bert_tokenizer(token_sequence, return_tensors="pt", truncation = True)
            outputs = bert_model(**inputs)
            features = outputs.pooler_output.detach().numpy()

            return features

    # define LayoutXLM (BERT-based) feature extractor pretrained on a multilingual colection of documents
    # https://huggingface.co/microsoft/layoutxlm-base
    elif extractor == 'bert_layoutxlm':
        from transformers import LayoutXLMProcessor, LayoutLMv2Model
        from doc2data.experimental.task_processors import TokenClsProcessorLayoutXLM
        from doc2data.utils import denormalize_bbox

        layoutxlm_processor = LayoutXLMProcessor.from_pretrained("microsoft/layoutxlm-base", apply_ocr=False)
        layoutxlm_processor.tokenizer.model_max_length = 512
        layoutxlm_model = LayoutLMv2Model.from_pretrained("microsoft/layoutxlm-base")

        def feature_extractor(token_dict, image):

            bboxes = [denormalize_bbox(i['bbox'], image.width, image.height) for i  in token_dict]
            bboxes = [TokenClsProcessorLayoutXLM.normalize_bbox_layoutxlm(i, image.width, image.height) for i in bboxes]
            tokens = [i['text'] for i  in token_dict]

            inputs = layoutxlm_processor(images = image, text = tokens, boxes = bboxes, padding="max_length", return_tensors = 'pt', truncation=True)
            outputs = layoutxlm_model(**inputs)
            features = outputs.pooler_output.detach().numpy()

            return features

    return feature_extractor

# function to extract features from images of individual document pages
def extract_features(pdf_collection, output_path, extractor_type):

    # prepare output folder
    os.makedirs(output_path, exist_ok = False)  # do not overwrite existing data

    # create_feature extractor
    feature_extractor = create_feature_extractor(extractor_type)

    # get paths for processed documents
    path_df = pdf_collection.image_path_df

    # run feature extraction for VGG-based extractors
    if 'vgg' in extractor_type:
        for i, row in tqdm(path_df.iterrows(), total = len(path_df)):

            image = load_image(row.image_path, target_size = (224, 224), to_array = False)
            features = feature_extractor(image)

            filename = os.path.splitext(os.path.basename(row.image_path))[0] + '.npy'
            np.save(os.path.join(output_path, filename), features)

    # run feature extraction for BERT-based extractor
    elif 'german' in extractor_type:
        for i, row in tqdm(path_df.iterrows(), total = len(path_df)):

            with open(row.tokens_path, 'rt') as file:
                token_dict = json.load(file)
            features = feature_extractor(token_dict)

            filename = os.path.splitext(os.path.basename(row.tokens_path))[0] + '.npy'
            np.save(os.path.join(output_path, filename), features)

    # run feature extraction for LayoutXLM-based extractor
    elif 'layoutxlm' in extractor_type:
        # run feature extraction
        for i, row in tqdm(path_df.iterrows(), total = len(path_df)):

            with open(row.tokens_path, 'rt') as file:
                token_dict = json.load(file)
            image = load_image(row.image_path, to_array = False)
            features = feature_extractor(token_dict, image)

            filename = os.path.splitext(os.path.basename(row.tokens_path))[0] + '.npy'
            np.save(os.path.join(output_path, filename), features)

    print(f'features extracted from {len(path_df)} documents with {extractor_type} extractor')


# load collection
pdf_collection = PDFCollection.load('processed_data/pdf_collection.pickle')

# extract imagenet features
extract_features(
    pdf_collection = pdf_collection,
    output_path = 'processed_data/features/vgg_imagenet',
    extractor_type = 'vgg_imagenet'
)

# extract rvl features
extract_features(
    pdf_collection = pdf_collection,
    output_path = 'processed_data/features/vgg_rvl',
    extractor_type = 'vgg_rvl'
)

# extract bert features
extract_features(
    pdf_collection = pdf_collection,
    output_path = 'processed_data/features/bert_german',
    extractor_type = 'bert_german'
)

# extract layoutxlm features
extract_features(
    pdf_collection = pdf_collection,
    output_path = 'processed_data/features/bert_layoutxlm',
    extractor_type = 'bert_layoutxlm'
)
