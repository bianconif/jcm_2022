"""Main script"""
import os
import pandas as pd
import numpy as np
from tabulate import tabulate

from cenotaph.classification.one_class import NND, NNPC
from cenotaph.colour.colour_descriptors import FullHist, Percentiles
from cenotaph.texture.hep.greyscale import LBP
from cenotaph.texture.hep.colour import OCLBP
#from cenotaph.cnn import ResNet50

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

#Folder where to store the results ion LaTeX form
latex_folder = 'LaTeX'

#Fraction of normal samples used for training the classifier
train_ratio = 0.5

#Number of train/test splits
num_splits = 10

#Create the cache folders if they do not exist
dirs = [classification_cache, feature_cache, splits_cache, latex_folder]
for dir_ in dirs:
    if not os.path.isdir(dir_):
        os.makedirs(dir_)

#descriptors = {'ColourHist': FullHist(nbins = 10),
               #'Percentiles': Percentiles(),
               #'LBP': LBP(),
               #'ResNet-50': ResNet50()}
descriptors = {'ColourHist': FullHist(nbins = 10),
               'Percentiles': Percentiles(),
               'OCLBP': OCLBP(),
               'LBP': LBP()}
classifiers = {'1-NN': NND(), 'NNPC': NNPC()}
datasets = ['Concrete-01', 'Fabric-01', 'Paper-01', 'Paper-02', 'Paper-03']

for classifier_name, classifier in classifiers.items():
    
    df = pd.DataFrame()
    
    for dataset in datasets:
    
        source_folder = f'{data_folder}/{dataset}'
        
        record = dict()
    
        for descriptor_name, descriptor in descriptors.items():     
            accuracy =\
                get_accuracy(descriptor = descriptor, descriptor_name = descriptor_name, 
                             classifier = classifier, classifier_name = classifier_name,
                             dataset_folder = source_folder, dataset_name = dataset,
                             feature_cache = feature_cache, 
                             classification_cache = classification_cache,
                             splits_cache = splits_cache,
                             train_ratio = train_ratio, num_splits = num_splits) 
            
            avg_acc = 100*np.mean(accuracy)
            print(f'Avg accuracy of {descriptor_name}/{classifier_name} on '
                  f'{dataset} = {avg_acc:4.2f}')
            
            record.update({'Feature': descriptor_name,
                           'Dataset': dataset,
                           'Accuracy (mean)': avg_acc})
            
            df = df.append(record, ignore_index=True)
    
    print(f'Classifier: {classifier_name}')
    print(tabulate(df))
    
    #---------- Store the results in a LaTeX table ----------
    latex_dest = f'{latex_folder}/{classifier_name}.tex'
    with open(latex_dest, 'w') as fp:
        #Header 
        cols = (['c']*(len(datasets) + 1))
        cols = ''.join(cols)
        fp.write(f'\\begin{{tabular}}{{{cols}}}\n')
        fp.write(f'\\toprule\n')
        fp.write(f'& \\multicolumn{{{len(datasets)}}}{{c}}{{Datasets}}\\\\')
        fp.write('Descriptor')
        offset = ord('A')
        
        str_ = str()
        for d, _ in enumerate(datasets):
            fp.write(f' & {chr(offset + d)}')
        fp.write('\\\\\n')
        fp.write(f'\\midrule\n')
        
        #Records
        for descriptor_name in descriptors.keys():
            fp.write(f'{descriptor_name}')
            for dataset in datasets:
                acc = df.loc[(df['Feature'] == descriptor_name) &
                             (df['Dataset'] == dataset)]['Accuracy (mean)']
                acc = acc.tolist()[0]
                fp.write(f' & {acc:3.1f}')
            fp.write('\\\\\n')
        
        #Footer
        fp.write(f'\\bottomrule\n')
        fp.write(f'\\end{{tabular}}\n')
    #--------------------------------------------------------
            
            