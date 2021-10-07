"""Main script"""
import os

from cenotaph.classification.one_class import SVDD
from cenotaph.colour.colour_descriptors import Percentiles

from functions import FeatureCalculator, SplitsGenerator

#Base folder for the image datasets
data_folder = '../data/images'

#Cache folder where to store image features and splits
cache_folder = '../cache'
feature_cache = f'{cache_folder}/features'

#Fraction of normal samples used for training the classifier
train_ratio = 0.5

#Number of train/test splits
num_splits = 10

#Create the cache folders if they do not exist
dirs = [feature_cache]
for dir_ in dirs:
    if not os.path.isdir(dir_):
        os.makedirs(dir_)

descriptors = {'Percentiles': Percentiles()}
datasets = ['Concrete-01', 'Paper-01']

splits_generator = SplitsGenerator()

for dataset in datasets:
    
    source_folder = f'{data_folder}/{dataset}'
    
    for descriptor_name, descriptor in descriptors.items():
    
        #Compute/read the features and the labels
        cache = f'{feature_cache}/{dataset}--{descriptor_name}.csv'
        feature_calculator = FeatureCalculator(descriptor, source_folder,
                                               cache = cache)
        features, labels = feature_calculator.get_features()
        
        #Generate the splits fro accuracy estimation
        train_indices = splits_generator._generate_splits(
            labels, train_ratio, num_splits)
        
        a = 0
