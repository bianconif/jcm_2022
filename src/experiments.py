"""Main script"""
import os
import numpy as np
import tensorflow as tf

from cenotaph.classification.one_class import NND, SVDD
from cenotaph.colour.colour_descriptors import Percentiles
from cenotaph.cnn import ResNet50

from functions import get_accuracy

#This is to avoid memory errors with convnets
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

#Base folder for the image datasets
data_folder = '../data/images'

#Cache folder where to store image features and splits
cache_folder = '../cache'
feature_cache = f'{cache_folder}/features'

#Cache folder where to store the train/test splits
splits_cache = f'{cache_folder}/splits'

#Cache folder where to store the classification results
classification_cache = f'{cache_folder}/classification'

#Fraction of normal samples used for training the classifier
train_ratio = 0.5

#Number of train/test splits
num_splits = 10

#Create the cache folders if they do not exist
dirs = [classification_cache, feature_cache, splits_cache]
for dir_ in dirs:
    if not os.path.isdir(dir_):
        os.makedirs(dir_)

descriptors = {'Percentiles': Percentiles(),
               'ResNet-50': ResNet50()}
classifiers = {'1-NN': NND()}
datasets = ['Paper-01', 'Paper-02', 'Paper-03']

for dataset in datasets:
    
    source_folder = f'{data_folder}/{dataset}'
    
    for descriptor_name, descriptor in descriptors.items():
        for classifier_name, classifier in classifiers.items():
            accuracy =\
                get_accuracy(descriptor = descriptor, descriptor_name = descriptor_name, 
                             classifier = classifier, classifier_name = classifier_name,
                             dataset_folder = source_folder, dataset_name = dataset,
                             feature_cache = feature_cache, 
                             classification_cache = classification_cache,
                             splits_cache = splits_cache,
                             train_ratio = train_ratio, num_splits = num_splits) 
            
            print(f'Avg accuracy of {descriptor_name}/{classifier_name} on '
                  f'{dataset} = {100*np.mean(accuracy):4.2f}')