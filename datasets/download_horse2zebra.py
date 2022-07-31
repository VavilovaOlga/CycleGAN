import os
from zipfile import ZipFile

os.system("pip install wget")
os.system("python -m wget https://people.eecs.berkeley.edu/~taesung_park/CycleGAN/datasets/horse2zebra.zip")
with ZipFile('horse2zebra.zip', 'r') as zip_file:
    zip_file.extractall()
os.remove('horse2zebra.zip')
