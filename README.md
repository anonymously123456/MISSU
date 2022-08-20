# MISSU
# Requirements
python 3.7
pytorch 1.6.0
torchvision 0.7.0
pickle
nibabel
# Data Acquisition
The BraTS 2019 could be acquired from here: https://ipp.cbica.upenn.edu/jobs/104051508197051009.

The liver tumor dataset CHAOS could be acquired from here: https://zenodo.org/record/3431873#.YwFeQi_4jUp.
# Data Preprocess
You shold download dataset and put data under /data directory, all data should be transform to .pkl.
# Training
Please using following commands to train a model.

python train.py
# Testing

python test.py

After the testing process stops, you can upload the submission file here: https://ipp.cbica.upenn.edu/.
