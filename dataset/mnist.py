import os
import idx2numpy
from torch.utils.data import Dataset
import numpy as np
import cv2

class Mnist(Dataset):
    """
    load mnist dataset from local directory
    """
    def __init__(self, root, args, folder):
        super().__init__()
        xtr_file = os.path.join(root, 'train-images-idx3-ubyte')
        ytr_file = os.path.join(root, 'train-labels-idx1-ubyte')
        xte_file = os.path.join(root, 't10k-images-idx3-ubyte')
        yte_file = os.path.join(root, 't10k-labels-idx1-ubyte')
        xtr = idx2numpy.convert_from_file(xtr_file)
        xtr = np.expand_dims(xtr, axis=1).astype(np.float32)
        ytr = idx2numpy.convert_from_file(ytr_file).astype(np.int64)
        xte = idx2numpy.convert_from_file(xte_file)
        xte = np.expand_dims(xte, axis=1).astype(np.float32)
        yte = idx2numpy.convert_from_file(yte_file).astype(np.int64)

        if folder == 'val':
            self.data = xte
            self.label = yte
        else:
            self.data = xtr
            self.label = ytr
        print('{} folder image and label shape:'.format(folder), self.data.shape, self.label.shape)

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, index):
        return self.data[index], self.label[index]
    
class MNIST_silo(Dataset):
    """
    load mnist dataset from local directory, with silo partition
    """
    def __init__(self, root, args, folder, silo):
        super().__init__()
        self.noise = args.noise
        self.num_silo = args.num_silo
        dataset_dir = os.path.join(root, 'MNIST/raw')
        xtr_file = os.path.join(dataset_dir, 'train-images-idx3-ubyte')
        ytr_file = os.path.join(dataset_dir, 'train-labels-idx1-ubyte')
        xte_file = os.path.join(dataset_dir, 't10k-images-idx3-ubyte')
        yte_file = os.path.join(dataset_dir, 't10k-labels-idx1-ubyte')

        # partition the dataset into silos
        xtr = idx2numpy.convert_from_file(xtr_file)[silo*(args.sample):(silo+1)*(args.sample)].astype(np.float32)
        ytr = idx2numpy.convert_from_file(ytr_file)[silo*(args.sample):(silo+1)*(args.sample)].astype(np.int64)
        xte = idx2numpy.convert_from_file(xte_file)[silo*(args.sample//4):(silo+1)*(args.sample//4)].astype(np.float32)
        yte = idx2numpy.convert_from_file(yte_file)[silo*(args.sample//4):(silo+1)*(args.sample//4)].astype(np.int64)

        if folder == 'val':
            self.data = self.transform(xte, silo)
            self.label = yte
        else:
            self.data = self.transform(xtr, silo)
            self.label = ytr
        print('{} folder image and label shape:'.format(folder), self.data.shape, self.label.shape)
    
    # rotate the image
    def augment_silos(self, x, silo, num_silo):
        angles = [0, 0, 0, -50, 120, 0, -65] if num_silo == 5 else [0, 0, 0, -50, -50, -50, 120, 120, 120]
        
        angle = angles[silo]
        if len(x.shape) == 3:
            h, w, c = x.shape
        else:
            h, w = x.shape
        center = (w / 2, h / 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
    
        return cv2.warpAffine(x, M, (w, h))

    def transform(self, x, silo):
        rotate_x = []
        for i in range(x.shape[0]):
            rotate_x.append(self.augment_silos(x[i], silo, self.num_silo))
        rotate_x = np.asarray(rotate_x)
        rotate_x = np.expand_dims(rotate_x, axis=1)
        return rotate_x
    
    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, index):
        return self.data[index], self.label[index]