import os
import shutil
from zipfile import ZipFile

os.system("pip install gdown")
os.system("gdown 1KWPc4Pa7u2TWekUvNu9rTSO0U2eOlZA9")

with ZipFile('faces_dataset_small.zip', 'r') as zip_file:
    zip_file.extractall()
shutil.rmtree('faces_dataset_small/04000')
shutil.rmtree('__MACOSX')
os.remove('faces_dataset_small/.DS_Store')
os.remove('faces_dataset_small.zip')
