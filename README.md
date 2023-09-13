# About
This repository contains the code for the article "Utilizing the omnipresent: Incorporating digital documents into predictive process monitoring using deep neural networks", published in Decision Support Systems.

**Article DOI: [10.1016/j.dss.2023.114043](https://doi.org/10.1016/j.dss.2023.114043)**

**Dataset DOI: [10.17632/kdcspz6xtn.1](https://doi.org/10.17632/kdcspz6xtn.1)**

The code is set up as a collection of scripts that need to be executed in sequential order.

* **01_document_processing.py:** Uses the library ```doc2data``` to parse PDF documents, extract page images, and apply OCR to extract text and bounding boxes where necessary.
* **02_feature_extraction.py:** Applies pretrained neural networks as feature extractors to obtain meaningful representations of each document page as context information of the event log.
* **03_dataset_compilation.py**: Creates tensorflow datasets from the event log augmented by the extracted features for each feature type and data split.
* **04_model_training.py:** Trains neural networks as PPM models for each type of extracted features. The models are trained for a grid of hyperparameters and for each data split.
* **05_test_set_evaluation.py:** Selects the best model from hyperparameter tuning based on the validation set and evaluates the model on the test set for each data split.
* **06_evaluation.R:** Computes performance metrics and visualizes training process.
* **07_embedding_extraction.py**: Extracts fixed-size embeddings from the integration module for each multi-page document.
* **08_shap_values.py:** Computes SHAP values for individual predictions of the damage type model.
* **09_explainability.R:** Runs explainability analysis based on the calculated document embeddings and SHAP values.
* **10_export.R:** Produces data for all tables and figures in the associated article, where applicable.

# Usage
This code was tested on a system running Ubuntu 22.04. It requires a machine with 64GB RAM and an Nvidia GPU with at least 4GB of VRAM. To reproduce the results please follow the outlined steps:

## 1. Clone repo
```
git clone https://github.com/serge724/ppm_docs.git
cd ppm_docs
```

## 2. Download datasets
You can access and download the required datasets from [Mendeley Data](https://data.mendeley.com/datasets/kdcspz6xtn/1).

We provide a shell script to create the necessary directories and automate the download. To run it, use the following command:

```
bash download_data.sh
```

In case you need to download the files manually, ensure they are placed as follows:
* log_data/process_log.csv
* log_data/folds_and_splits.csv
* processed_data/features/bert_german.zip
* processed_data/features/bert_layoutxlm.zip
* processed_data/features/vgg_imagenet.zip
* processed_data/features/vgg_rvl.zip

All zip files need to be unzipped before continuing.

## 3. Create conda environment
```
conda env create -f conda_env.yml
conda activate ppm_docs
```

## 4. Install R libraries
The R scripts rely on the following libraries:
```
readr 2.1.2
dplyr 1.0.9
tidyr 1.2.0
stringr 1.4.0
purrr 0.3.4
magrittr 2.0.3
ggplot2 3.3.6
caret 6.0-93
bupaR 0.5.2
```
While the R scripts may work with different versions of the listed libraries, this was not tested.

## 5. Run scripts

In order to reproduce our results, you can run the scripts sequentially from the terminal:

```
python 03_dataset_compilation.py
python 04_model_training.py
python 05_test_set_evaluation.py
Rscript 06_evaluation.R
python 07_embedding_extraction.py
python 08_shap_values.py
Rscript 09_explainability.R
Rscript 10_export.R
```

The final results that are also reported in the paper are saved to the directory ```results/export/```. Please note that random initialization and a different computational environment may result in slight deviations.
