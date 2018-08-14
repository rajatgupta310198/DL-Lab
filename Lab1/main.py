from dataset import CatDog
import tensorflow as tf
data = CatDog()

X_train_images, X_train_labels = data.train_images, data.train_labels

print(X_train_images.shape, X_train_labels.shape)