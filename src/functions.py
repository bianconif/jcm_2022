import os
import pandas as pd

import numpy as np

from cenotaph.basics.base_classes import Image
from cenotaph.basics.generic_functions import get_files_in_folder, get_folders

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
        