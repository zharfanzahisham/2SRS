import os
import numpy as np
from scipy.io import loadmat
from keras import layers
from keras.models import Model


class Data:
    @staticmethod
    def load_data(name):
        dataset_path = './datasets/benchmark/'
        if name == 'IP':
            data = loadmat(os.path.join(dataset_path, 'Indian_pines_corrected.mat'))['indian_pines_corrected']
            labels = loadmat(os.path.join(dataset_path, 'Indian_pines_gt.mat'))['indian_pines_gt']  
            n_classes = 17
            dataset_name = 'indian_pines'
            rgb_bands = (29, 20, 11)
        elif name == 'SA':
            data = loadmat(os.path.join(dataset_path, 'Salinas_corrected.mat'))['salinas_corrected']
            labels = loadmat(os.path.join(dataset_path, 'Salinas_gt.mat'))['salinas_gt']
            n_classes = 17
            dataset_name = 'salinas_scene'
            rgb_bands = (29, 20, 11)
        else:  # PU dataset
            data = loadmat(os.path.join(dataset_path, 'PaviaU.mat'))['paviaU']
            labels = loadmat(os.path.join(dataset_path, 'PaviaU_gt.mat'))['paviaU_gt']
            n_classes = 10
            dataset_name = 'pavia_university'
            rgb_bands = (29, 20, 11)
            
        return data, labels, n_classes, dataset_name, rgb_bands


class PreProcessing:
    @staticmethod
    def create_image_cubes(X, y, window_size=25, include_zero_labels=True):
        if window_size%2 == 0:
            margin = int((window_size) / 2)
        else:
            margin = int((window_size - 1) / 2)

        # Pad the data with zeros
        zero_padded_X = np.pad(X, [(margin, margin), (margin, margin), (0, 0)])

        # Create image patches with 0 values first
        patches_data = np.zeros((X.shape[0] * X.shape[1], window_size, window_size, X.shape[2]))
        patches_label = np.zeros((X.shape[0] * X.shape[1]))
        
        # Assign values to the image patches
        patch_index = 0
        for r in range(margin, zero_padded_X.shape[0] - margin):
            for c in range(margin, zero_padded_X.shape[1] - margin):
                if window_size%2 == 0:
                    patch = zero_padded_X[r - margin:r + margin, c - margin:c + margin]
                else:
                    patch = zero_padded_X[r - margin:r + margin + 1, c - margin:c + margin + 1]
                patches_data[patch_index, :, :, :] = patch
                patches_label[patch_index] = y[r-margin, c-margin]
                patch_index = patch_index + 1

        # Remove zero labels
        if not include_zero_labels:
            patches_data = patches_data[patches_label>0,:,:,:]
            patches_label = patches_label[patches_label>0]
            patches_label -= 1
        
        return patches_data, patches_label

    @staticmethod
    def create_image_cubes2(data, labels=None, window_size=25, include_zero_labels=True):
        original_shape = data.shape
        # Perform zero-padding to the data (only across the spatial dimension, which means the channels are not padded)
        margin = int((window_size - 1) / 2)
        data = np.pad(data, [(margin, margin), (margin, margin), (0, 0)])
        print(f'Shape after padding with zeros: {data.shape}')
        
        # Initialize patches of data and labels with zero values
        data_patches = np.zeros((original_shape[0] * original_shape[1], window_size, window_size, original_shape[2]))
        if labels is not None:
            label_patches = np.zeros((original_shape[0] * original_shape[1]))
        
        # Crop patches from original data and assign to the data_patches and label_patches
        for i in range(original_shape[0]):
            for j in range(original_shape[1]):
                patch = data[i:i+window_size, j:j+window_size]
                data_patches[i*original_shape[0]+j, :, :, :] = patch
                if labels is not None:
                    label_patches[i*original_shape[0]+j] = labels[i, j]
        
        # Remove the pixels where the label is zero
        if not include_zero_labels:
            data_patches = data_patches[label_patches > 0, :, :, :]
            label_patches = label_patches[label_patches > 0]
            label_patches -= 1
        
        if labels is not None:    
            return data_patches, label_patches
        return data_patches
