# Author: Sergej Levich
# Journal article: Sergej Levich et al., Decision Support Systems, https://doi.org/10.1016/j.dss.2023.114043

import os
import json
import logging
import pandas as pd
from PIL import Image
from tqdm import tqdm
from doc2data.pdf import PDFCollection
from doc2data.experimental.preprocessing import Extractor
from doc2data.experimental.ocr import run_doctr_ocr

logging.basicConfig(level=logging.INFO)

# # create PDF collection
pdf_collection = PDFCollection('log_data/documents/')

# parse files
pdf_collection.parse_files()

# instantiate extractor
extractor = Extractor()

# extract images
extractor.extract_page_images_from_pdfs(
    pdf_collection = pdf_collection,
    target_dir = 'processed_data/images',
    format = 'jpg',
    dpi = 100,
    overwrite = False
)

# extract tokens
os.makedirs('processed_data/tokens', exist_ok = True)
tokens_path = []
for source_path in tqdm(pdf_collection.image_path_df.image_path):
    while True:
        try:
            image = Image.open(source_path)
            tokens = run_doctr_ocr(image)
            target_path = f'processed_data/tokens/{os.path.splitext(os.path.basename(source_path))[0]}.json'
            with open(target_path, 'wt') as file:
                json.dump(tokens, file)
            tokens_path.append(target_path)
            break
        except KeyboardInterrupt:
            raise KeyboardInterrupt
        except:
            print(f'OCR failed for {source_path}, repeating attempt...')
            pass

pdf_collection.image_path_df['tokens_path'] = tokens_path

# save collection
pdf_collection.save('processed_data/pdf_collection.pickle')
