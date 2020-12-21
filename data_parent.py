from PIL import Image
import os
import numpy as np
import sys


class load_data_parent:
    def __init__(self, train_list, database_root):
        """ Load data for parent network
        train_list: list with the paths of the images to use for training
        database_root: Path to the root of the Database
        """
        # Load training images (path) and labels
        if not isinstance(train_list, list) and train_list is not None:
            with open(train_list) as t:
                train_paths = t.readlines()
        elif isinstance(train_list, list):
            train_paths = train_list
        else:
            train_paths = []

        self.images_train = []
        self.images_train_path = []
        self.labels_train = []
        self.labels_train_path = []

        # Start loading
        print('Start loading files:')
        for idx, line in enumerate(train_paths):
            img = Image.open(os.path.join(database_root, str(line.split()[0])))
            img.load()
            label = Image.open(os.path.join(database_root, str(line.split()[1])))
            label.load()
            label = label.split()[0]

            if idx == 0: sys.stdout.write('Loading the data')
            self.images_train.append(np.array(img, dtype=np.uint8))
            self.labels_train.append(np.array(label, dtype=np.uint8))

            if (idx + 1) % 50 == 0:
                sys.stdout.write('.')
            self.images_train_path.append(os.path.join(database_root, str(line.split()[0])))
            self.labels_train_path.append(os.path.join(database_root, str(line.split()[1])))
        sys.stdout.write('\n')
        self.images_train_path = np.array(self.images_train_path)
        self.labels_train_path = np.array(self.labels_train_path)

        print('Finish loading Dataset')

        # Init parameters
        self.train_ptr = 0
        self.test_ptr = 0
        self.train_size = max(len(self.images_train_path), len(self.images_train))
        self.train_idx = np.arange(self.train_size)
        np.random.shuffle(self.train_idx)
        self.store_memory = True  # stores all the training images

    def next_batch_parent(self, batch_size):
        """Get next batch of image (path) and labels
        batch_size: Size of the batch
        Returns:
        images: List of images paths if store_memory=False, List of Numpy arrays of the images if store_memory=True
        labels: List of labels paths if store_memory=False, List of Numpy arrays of the labels if store_memory=True
        """
        if self.train_ptr + batch_size < self.train_size:
            idx = np.array(self.train_idx[self.train_ptr:self.train_ptr + batch_size])
            if self.store_memory:
                images = [self.images_train[l] for l in idx]
                labels = [self.labels_train[l] for l in idx]
            else:
                images = [self.images_train_path[l] for l in idx]
                labels = [self.labels_train_path[l] for l in idx]
            self.train_ptr += batch_size
        else:
            old_idx = np.array(self.train_idx[self.train_ptr:])
            np.random.shuffle(self.train_idx)
            new_ptr = (self.train_ptr + batch_size) % self.train_size
            idx = np.array(self.train_idx[:new_ptr])
            if self.store_memory:
                images_1 = [self.images_train[l] for l in old_idx]
                labels_1 = [self.labels_train[l] for l in old_idx]
                images_2 = [self.images_train[l] for l in idx]
                labels_2 = [self.labels_train[l] for l in idx]
            else:
                images_1 = [self.images_train_path[l] for l in old_idx]
                labels_1 = [self.labels_train_path[l] for l in old_idx]
                images_2 = [self.images_train_path[l] for l in idx]
                labels_2 = [self.labels_train_path[l] for l in idx]
            images = images_1 + images_2
            labels = labels_1 + labels_2
            self.train_ptr = new_ptr
        return images, labels

    def get_train_size(self):
        return self.train_size

    def train_img_size(self):
        width, height = Image.open(self.images_train[self.train_ptr]).size
        return height, width
