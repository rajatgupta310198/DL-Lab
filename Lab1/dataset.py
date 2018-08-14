import cv2
import numpy as np
import os, re

# Cat 0
# Dog 1

class CatDog(object):

    def __init__(self, dir_name='/dataset'):
        self.data_dir = dir_name
        self.train_images, self.train_labels, self.test_images, self.test_labels  = self.read_train_images()
        
        print(self.train_images.shape, self.train_labels.shape)
        self.current_index = 0

    def read_train_images(self):
        train_images = os.listdir(os.getcwd() + self.data_dir +'/train')
        train_images_list = []
        train_labels_list = []
        print(len(train_images))
        for image in train_images:
            img = cv2.imread(os.getcwd() + self.data_dir +'/train/' + str(image))
            img = cv2.resize(img, (50, 50))
            train_images_list.append(img)
            if image == re.match('cat', image):
                train_labels_list.append(0)
            else:
                train_labels_list.append(1)
        return np.array(train_images_list[:23000]), self.one_hot(train_labels_list[:23000]), np.array(train_images_list[23000:]), self.one_hot(train_labels_list[23000:])
        
    def next_batch(self, batch_size):
        if self.current_index + batch_size > len(self.train_images):
            images, labels = self.train_images[self.current_index:], self.train_labels[self.current_index:]
            self.current_index = 0
            return images, labels
        else:
            images, labels = self.train_images[self.current_index : self.current_index + batch_size], self.train_labels[self.current_index: self.current_index+batch_size]
            self.current_index += batch_size
            return images, labels

        
    def one_hot(self, labels):
        labels = np.array(labels)
        b = np.zeros((len(labels), 2))
        b[np.arange(len(labels)), labels] = 1
        return b


    def read_test_images(self):
        test_images = os.listdir(os.getcwd() + self.data_dir +'/test1')
        test_images_list = []
        test_labels_list = []
        print(len(test_images))
        for image in test_images:
            img = cv2.imread(os.getcwd() + self.data_dir +'/test1/' + str(image))
            img = cv2.resize(img, (50, 50))
            test_images_list.append(img)
            
        return np.array(test_images_list)

    def get_test(self):

        return self.test_images, self.test_labels



