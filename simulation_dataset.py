import  os.path
import  numpy as np
import torch
import random
from torch.utils.data import Dataset, DataLoader
import io, requests

from datetime import datetime

import matplotlib.pyplot as plt

class SimulationDataset(Dataset):
    '''
    This class creates the custom PyTorch DataLoader class, loading the numpy dataset into memory
    '''
    def __init__(self, data_folder, local_data, total_sample_number = None, transform = None):

        #define dielectric constants of the air and substrate for data preprocessing
        n_air=1.
        n_sub=1.45

        print("Loading training data...")
        print("Note: The training and test datasets are very large, which can take "\
                   "on the order of hours to complete, depending on your internet download "\
                   "speed. If you have a slower internet download speed, consider downloading the "\
                   "data locally from http://metanet.stanford.edu/search/waveynet-study/ so "\
                   "that you do not have to re-download the data each time you run the code."\
                   "See the README file for more information.")
        #load the training data, either locally or from Metanet, as specified by
        #the boolean args.local_data
        if local_data:
            train_data = np.load(f'{data_folder}/train_ds.npz')
        else:
            response = requests.get('http://metanet.stanford.edu/static/search/waveynet/'\
                                        'data/train_ds.npz')
            response.raise_for_status()
            train_data = np.load(io.BytesIO(response.content))

        print("Loading test data...")

        #load the test data, either locally or from Metanet, as specified by
        #the boolean args.local_data
        if local_data:
            test_data = np.load(f'{data_folder}/test_ds.npz')
        else:
            response = requests.get('http://metanet.stanford.edu/static/search/waveynet/'\
                                        'data/test_ds.npz')
            response.raise_for_status()
            test_data = np.load(io.BytesIO(response.content))

        print("Finished loading the dataset into memory.")

        #Load the dielectric structures, corresponding fields, and the refractive
        #index vectors from the downloaded training dataset
        self.input_imgs = train_data['structures'].astype(np.float32, copy=False)
        self.Hy_forward = train_data['Hy_fields'].astype(np.float32, copy=False)
        self.refr_idx_vec = train_data['dielectric_permittivities']

        #Concatenate the data from the test dataset to the training dataset
        test_input_imgs = test_data['structures'].astype(np.float32, copy=False)
        self.input_imgs = np.concatenate((self.input_imgs,test_input_imgs),axis=0)

        test_Hy_forward = test_data['Hy_fields'].astype(np.float32, copy=False)
        self.Hy_forward = np.concatenate((self.Hy_forward,test_Hy_forward),axis=0)

        test_refr_idx_vec = test_data['dielectric_permittivities']
        self.refr_idx_vec = np.concatenate((self.refr_idx_vec,test_refr_idx_vec),axis=0)

        #Scale the pixel values of the images representing the dielectric images
        #from [1,max_dielectric_value] to [0,1] for input into the neural network
        die_change = (n_sub-n_air)/(np.sqrt(self.refr_idx_vec)-n_air)

        for dev in range(len(self.input_imgs)):
                self.input_imgs[dev][np.where(self.input_imgs[dev]==self.input_imgs[dev,0,0,0])]=\
                                         die_change[dev]
                self.input_imgs[dev]=(self.input_imgs[dev]*(np.sqrt(self.refr_idx_vec[dev]) - \
                                         n_air) + n_air)**2

        #Use only a subset of the dataset if specified to do so by args.total_sample_number
        if total_sample_number:
            random.seed(1234)
            indices = random.sample(list(range(self.input_imgs.shape[0])), total_sample_number)
            self.input_imgs = self.input_imgs[indices, :, :, :]
            self.Hy_forward = self.Hy_forward[indices, :, :, :]
            self.refr_idx_vec = self.refr_idx_vec[indices, :, :, :]

        print("input_imgs.shape: ", self.input_imgs.shape, self.input_imgs.dtype)
        print("Hy_forward.shape: ", self.Hy_forward.shape, self.Hy_forward.dtype)
        print("refr_idx_vec.shape: ", self.refr_idx_vec.shape, self.refr_idx_vec.dtype)
        self.transform = transform

    def __len__(self):
        return self.input_imgs.shape[0]

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        structure = self.input_imgs[idx, :, :, :]
        field = self.Hy_forward[idx, :, :, :]
        eps_distr = self.refr_idx_vec[idx, :, :, :]

        sample = {'structure': structure, 'field': field, 'eps_distr': eps_distr}

        if self.transform:
            sample = self.transform(sample)

        return sample
