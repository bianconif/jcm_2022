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

def get_accuracy(descriptor, descriptor_name, classifier, classifier_name,
                 dataset_folder, dataset_name, feature_cache, 
                 classification_cache, splits_cache, train_ratio, num_splits):
    """Read/estimate the accuracy of a combination feature/classifier on a 
    given dataset
    
    Parameters
    ----------
    descriptor: cenotaph.basics.base_classes.ImageDescriptor
        The image descriptor.
    descriptor_name: str
        The name used to identify this descriptor
    classifier: cenotaph.classification.one_class.OneClassClassifier
        The one-class classifier.
    classifier_name: str
        The name used to identify the classifier
    dataset_folder: str
        Path to the dataset folder. See FeatureCalculator documentation for
        the folder structure.
    dataset_name: str
        The name used to identify the dataset.
    feature_cache: str
        Path to the folder where the features are cached.
    classification_cache: str
        Path to the folder where the classification results are cached.
    splits_cache: str
        Path to the folder where the splits are cached
    train_ratio: float [0,1]
        The fraction of normal samples used to train the classifier.
    num_splits: int
        The number of train/test splits
    
    Returns
    -------
    accuracy: nparray of float (num_splits)
        The accuracy for each split.
    """
    
    accuracy = np.empty((0,0), dtype=np.float)
    
    #Check if the results are already stored; if so then read them, otherwise
    #compute
    source = f'{classification_cache}/{dataset_name}--{descriptor_name}--{classifier_name}.csv'
    try:
        df = pd.read_csv(source)
    except FileNotFoundError:
            
        splits_generator = SplitsGenerator(cache = f'{splits_cache}/{dataset_name}.csv')
        
        print(f'Computing {descriptor_name}/{classifier_name} on {dataset_name}')

        #Compute/read the features and the labels
        cache = f'{feature_cache}/{dataset_name}--{descriptor_name}.csv'
        feature_calculator = FeatureCalculator(descriptor, dataset_folder,
                                               cache = cache)
        features, labels = feature_calculator.get_features()

        #--------- Generate/read the splits for accuracy estimation ---------

        #Generate/read the train indices
        train_indices = splits_generator.get_train_indices(
            labels, train_ratio, num_splits)


        #--------------------------------------------------------------------

        #---------- Estimate the accuracy for each split --------------------
        for s in range(train_indices.shape[1]):
            train_indices_s = train_indices[:,s]
            
            #Obtain the test indices by difference
            test_indices_s = set(range(len(labels))) - set(train_indices_s)
            test_indices_s = np.array(list(test_indices_s), dtype=np.int)            
            
            #Train the classifier
            classifier.train(positive_patterns = features[train_indices_s,:])
            
            #Predict the response and compute the accuracy
            target_labels = labels[test_indices_s]
            predicted_labels = classifier.predict(features[test_indices_s,:])
            acc = np.sum(target_labels == predicted_labels)/len(target_labels)
            accuracy.append(acc)

        #--------------------------------------------------------------------
        
        #Save the results
        df = pd.DataFrame(data = accuracy)
        df.to_csv(source, index = False)
        
    return accuracy

class SplitsGenerator():
    """Define train/test splits for accuracy estimation"""
    
    def __init__(self, normal_label='Normal', **kwargs):
        """
        
        Parameters
        ----------
        normal_label: str
            The label that identifies a pattern as 'normal'
        cache: str (optional)
            The cache where the splits are to be stored. If given the splits
            are cached in the given file and not recomputed. Use this option
            to freeze the splits and obtain reproducible results.
        """
        self._normal_label = normal_label
        if 'cache' in kwargs.keys():
            self._cache = kwargs['cache']
            
    def get_train_indices(self, labels, train_ratio, num_splits):
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
        
        try:
            cache = self._cache
            try:
                #If caching requested try and read the file
                train_indices = self._read_train_indices(self._cache)
            except FileNotFoundError:
                #If the cache file doesn't exist then compute and store the
                #splits
                train_indices = self._generate_splits(
                    labels, train_ratio, num_splits)
                self._save_train_indices(train_indices, self._cache)
        except AttributeError:
            #If caching not requested just compute the features without
            #storing them
            train_indices = self._generate_splits(
                labels, train_ratio, num_splits)
            
        return train_indices
    
    def _read_train_indices(self, source):
        """Read the train indices from csv file
        
        Parameters
        ----------
        source: str 
            Pointer to the csv file where the train indices are stored

        Returns
        -------
        train_indices: nparray of int (N,D)
            The train indices; where N is the total number of train samples and D
            the number of splits (each column represents one split into
            train and test set).
        """
        
        df = pd.read_csv(source)
        train_indices = df.to_numpy()
        
        return train_indices        
        
    def _save_train_indices(self, train_indices, destination):
        """Save the train indices into a csv file
        
        Parameters
        ----------
        train_indices: nparray of int (see self._generate_splits)
        destination: str
            The csv file where the train indices are to be stored.
        """
        
        df = pd.DataFrame(train_indices)
        df.to_csv(destination, index = False)    
        
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
        