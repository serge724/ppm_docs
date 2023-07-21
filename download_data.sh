#!/bin/bash

set -e  # Exit script on any error

# Define necessary programs
REQUIRED_PROGRAMS=(wget unzip)

# Check if necessary programs are installed
for program in "${REQUIRED_PROGRAMS[@]}"; do
    if ! command -v "${program}" >/dev/null 2>&1; then
        echo "Error: ${program} is not installed." >&2
        exit 1
    fi
done

# Define base directories and files
BASE_DIR="$(pwd)"
LOG_DIR="${BASE_DIR}/log_data"
PROCESSED_DATA_DIR="${BASE_DIR}/processed_data/features"

# Create directories
mkdir -p "${LOG_DIR}"
mkdir -p "${PROCESSED_DATA_DIR}"

# Define file URLs and names
declare -A FILE_URLS=(
    ["${LOG_DIR}/process_log.csv"]="https://data.mendeley.com/public-files/datasets/kdcspz6xtn/files/f0d20ffd-4226-44d7-a298-0ed88eb787be/file_downloaded"
    ["${LOG_DIR}/folds_and_splits.csv"]="https://data.mendeley.com/public-files/datasets/kdcspz6xtn/files/cd339b01-cc8b-466c-838f-a71613f18e95/file_downloaded"
    ["${PROCESSED_DATA_DIR}/bert_german.zip"]="https://data.mendeley.com/public-files/datasets/kdcspz6xtn/files/5c19c8ec-33a1-41c3-a191-6dd0416bf284/file_downloaded"
    ["${PROCESSED_DATA_DIR}/bert_layoutxlm.zip"]="https://data.mendeley.com/public-files/datasets/kdcspz6xtn/files/78435b70-8e10-4273-b66e-2d007c026ebd/file_downloaded"
    ["${PROCESSED_DATA_DIR}/vgg_imagenet.zip"]="https://data.mendeley.com/public-files/datasets/kdcspz6xtn/files/e791ffaa-ecc4-4799-98dc-a40ff8fc2006/file_downloaded"
    ["${PROCESSED_DATA_DIR}/vgg_rvl.zip"]="https://data.mendeley.com/public-files/datasets/kdcspz6xtn/files/d502565e-83da-4217-a0be-de1a742cccfe/file_downloaded"
)

# Download data
for file in "${!FILE_URLS[@]}"; do
    wget -O "${file}" "${FILE_URLS[$file]}"
done

# Unzip files in PROCESSED_DATA_DIR
cd "${PROCESSED_DATA_DIR}"
for file in *.zip; do
    unzip "${file}"
    rm "${file}"
done
