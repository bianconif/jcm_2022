from math import comb
import os
import pandas as pd

import numpy as np

from cenotaph.basics.base_classes import Image
from cenotaph.basics.generic_functions import get_files_in_folder, get_folders

def _shuffle(x):
    """Shuffle returning a copy"""
    a = np.copy(x)
    np.random.shuffle(a)
    return a

class SplitsGenerator():
    """Define train/test splits for accuracy estimation"""
    
    def __init__(self, normal_label='Normal'):
        """
        
        Parameters
        ----------
        normal_label: str
            The label that identifies a pattern as 'normal'
        """
        self._normal_label = normal_label
        
    def _generate_splits(self, labels, train_ratio, num_splits):
        
        """
        Parameters
        ----------
        labels: ndarray of str
            The label of each pattern in the dataset.
        train_ratio: float [0,1]
            The fraction of normal samples used to train the classifier
        num_splits: int
            The total number of splits
            
        Returns
        -------
        train_indices: nparray of int (x,num_splits)
            Indices of the patterns used for train, where 
            x = floor(train_ratio * number of normal samples)  
        """
        
        #Indices of the normal cases
        normal_indices = np.argwhere(labels == self._normal_label)
        
        #Total number of normal cases
        total_normal = len(normal_indices)
        
        #Number of train samples
        num_train_samples = np.floor(total_normal * train_ratio).astype(np.int)
        
        #Upper bound to the number of splits (number of subsets with 
        #train_number elements from a set with total_normal elements)
        max_allowed_splits = comb(total_normal, num_train_samples)
        if num_splits > max_allowed_splits:
            raise Exception('Not enough train samples for the requested number of splits')
        
        #Generate the splits
        train_indices = _shuffle(normal_indices)
        train_indices = train_indices[0:num_train_samples]
        while train_indices.shape[1] < num_splits:
            
            #Generate a tentative random permutation
            tentative_train_indices = _shuffle(normal_indices)[0:num_train_samples]
            
            #Make sure the permutation is different than those already stored
            already_present = False
            for i in range(train_indices.shape[1]):
                if np.sum(tentative_train_indices - train_indices[:,i]) == 0:
                    already_present = True
                    break
            if not already_present:
                train_indices = np.hstack((train_indices, 
                                           tentative_train_indices))
            
        
        return train_indices
        

class FeatureCalculator():
    
    #Column header for labels in the dataframe (for saving/reading)
    _labels_header = 'Labels'
    
    def __init__(self, image_descriptor, source_folder, **kwargs):
        """
        Parameters
        ----------
        source_folder: str
            The folder where the source images are stored. It is assumed that any
            subfolder within the source folder represents one different class.
            The name of each subfolder is taken as the class label.
            
            *** Sample folder structure: ***
            
            source_folder
            |-class1
            |  |-class1_img1.jpg      "The mask images are in this folder."
            |  |-class1_img2.jpg
            |  |-class1_img2.jpg
            |  |...
            |-class2
            |  |-class2_img1.jpg      "The mask images are in this folder."
            |  |-class2_img2.jpg
            |  |-class2_img2.jpg
            |  |...
            |...
        
        image_descriptor: cenotaph.basics.base_classes.ImageDescriptor
            The image descriptor
        cache: str (optional)
            Path to the file where the feature and labels are temporarily stored.
            If the file exists the features and labels are read from the cache 
            instead of being recomputed.
        """
        self._image_descriptor = image_descriptor
        self._source_folder = source_folder
        if 'cache' in kwargs.keys():
            self._cache = kwargs['cache']
    
    def get_features(self):
        """Compute or read the features
        
        Returns
        ----------
        features: nparray of float (N,D)
            The features; where N is the total number of images and D the dimension
            of the feature space.
        labels: nparray of object (N)
            The class label of each feature vector
        """
    
        try:
            cache = self._cache
            try:
                #If caching requested try and read the file
                features, labels = self._read_features(self._cache)
            except FileNotFoundError:
                #If the cache file doesn't exist then compute and store the
                #features
                features, labels = self._compute_features()
                self._save_features(features, labels, self._cache)
        except AttributeError:
            #If caching not requested just compute the features without
            #storing them
            features, labels = self._compute_features()       
            
        return features, labels
            
    
    @classmethod
    def _save_features(cls, features, labels, destination):
        """Save features and class labels
    
        Parameters
        ----------
        features: nparray of float (N,D)
            The features; where N is the total number of images and D the dimension
            of the feature space.
        labels: nparray of object (N)
            The class label of each feature vector
        destination: str
            The csv file where the data are to be stored.
        """
    
        df = pd.DataFrame(features)
        df[cls._labels_header] = labels
        df.to_csv(destination)
    
    @classmethod
    def _read_features(cls, source):
        """Read features and class labels

        Parameters
        ----------
        source: str
            The csv file where the features and labels are stored.

        Returns
        ----------
        features: nparray of float (N,D)
            The features; where N is the total number of images and D the dimension
            of the feature space.
        labels: nparray of object (N)
            The class label of each feature vector
        """
        
        #Read the dataframe and get the labels first
        df = pd.read_csv(source)
        labels = df[cls._labels_header].to_numpy()
        
        #Now drop the index and labels column to get the features
        df.drop(labels = cls._labels_header, axis=1, inplace=True)
        features = df.to_numpy()[:,1::]

        return features, labels

    def _compute_features(self):
        """Compute the image features
            
        Returns
        -------
        features: nparray of float (N,D)
            The features; where N is the total number of images and D the dimension
            of the feature space.
        labels: nparray of object (N)
            The class label of each image
            """
    
        features = np.empty((0), dtype=np.float)
        labels = np.empty((0), dtype=object)
        num_processed_images = 0
    
        folders = get_folders(self._source_folder)
        for sub_folder in folders:
        
            #Compute the features for each image
            images = get_files_in_folder(f'{self._source_folder}/{sub_folder}')
            for image in images:
                
                #Compute and append the feature vector
                fvector = self._image_descriptor.get_features(Image(image))
                fvector = np.expand_dims(fvector, axis=0)
                features = np.append(features, fvector)
                
                #Append the label
                labels = np.append(labels, sub_folder)
            
                num_processed_images += 1
    
            #Reshape the feature vector
            cols = len(features) // num_processed_images
            features = np.reshape(features, 
                                  newshape = (num_processed_images, cols))
        
        return features, labels
        