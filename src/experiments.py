"""Main script"""
import os
import pandas as pd
import numpy as np
from tabulate import tabulate

from cenotaph.basics.base_classes import Ensemble
from cenotaph.classification.one_class import EllipticEnvelope, NND, SVM
from cenotaph.colour.colour_descriptors import FullHist, MarginalHists
from cenotaph.texture.hep.greyscale import ILBP, LBP
from cenotaph.texture.hep.colour import OCLBP
from cenotaph.texture.filtering import Gabor
from cenotaph.cnn import MobileNet, ResNet50

from functions import get_accuracy

#This is to avoid memory errors with convnets
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

#Base folder for the image datasets
data_folder = '../data/images'

#Cache folder where to store image features and splits
cache_folder = '../cache'
feature_cache = f'{cache_folder}/features'
hep_luts = f'{cache_folder}/hep-luts'

#Cache folder where to store the classification results
classification_cache = f'{cache_folder}/classification'

#Folder where to store the results ion LaTeX form
latex_folder = 'LaTeX'

#Fraction of normal samples used for training the classifier
train_ratio = 0.5

#Number of train/test splits
num_splits = 50

#Cache folder where to store the train/test splits
splits_cache = f'{cache_folder}/splits/{num_splits}'

#Create the cache folders if they do not exist
dirs = [classification_cache, feature_cache, splits_cache, latex_folder,
        hep_luts]
for dir_ in dirs:
    if not os.path.isdir(dir_):
        os.makedirs(dir_)

#descriptors = {'ColourHist': FullHist(nbins = 10),
               #'Percentiles': Percentiles(),
               #'LBP': LBP(),
               #'ResNet-50': ResNet50()}

#Common settings for LBP-like descriptors
lbp_common_settings = {'num_peripheral_points': 8, 'group_action': 'C',
                       'cache_folder': hep_luts}      
               
traditional_descriptors =\
    {'FullColHist': FullHist(nbins = 10),
     'MargColHists': MarginalHists(nbins = (256, 256, 256)),
     'Gabor': Gabor(size = 6),
     'LBP': Ensemble(image_descriptors=
                     [LBP(radius=1, **lbp_common_settings),
                      LBP(radius=2, **lbp_common_settings),
                      LBP(radius=3, **lbp_common_settings)]),
     'ILBP': Ensemble(image_descriptors=
                     [ILBP(radius=1, **lbp_common_settings),
                      ILBP(radius=2, **lbp_common_settings),
                      ILBP(radius=3, **lbp_common_settings)]),      
     'OCLBP': Ensemble(image_descriptors=
                     [OCLBP(radius=1, **lbp_common_settings),
                      OCLBP(radius=2, **lbp_common_settings),
                      OCLBP(radius=3, **lbp_common_settings)])
     }
cnns = {'MobileNet': MobileNet(), 'ResNet-50': ResNet50()}
descriptors = {**traditional_descriptors, **cnns}

classifiers = {'3-NN': NND(k = 3)}
datasets = ['Carpet-01', 'Concrete-01', 'Fabric-01', 'Fabric-02', 'Layered-01',
            'Leather-01', 'Paper-01', 'Paper-02', 'Wood-01']

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
        #for d, _ in enumerate(datasets):
            #fp.write(f' & {chr(offset + d)}')
            
        for dataset in datasets:
            fp.write(f' & {dataset}')        
        
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
            
            