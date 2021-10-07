"""Main script"""
import os

from cenotaph.colour.colour_descriptors import Percentiles

#Cache folder where to store image features and splits
cache_folder = '../cache'
feature_cache = f'{cache_folder}/features'

#Create the cache folders if they do not exist
dirs = [feature_cache]
for dir_ in dirs:
    if not os.path.isdir(path)

descriptors = {'Percentiles': Percentiles()}
datasets = ['Concrete', 'Paper-1']
