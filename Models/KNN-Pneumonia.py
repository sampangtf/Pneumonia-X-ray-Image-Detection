#!/usr/bin/env python
# coding: utf-8

# In[1]:


# import the necessary packages
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from imutils import paths
import numpy as np
import argparse
import imutils
import cv2
import os


# In[2]:


def image_to_feature_vector(image, size=(32, 32)):
	# resize the image to a fixed size, then flatten the image into
	# a list of raw pixel intensities
	return cv2.resize(image, size).flatten()


# In[3]:


def extract_color_histogram(image, bins=(8, 8, 8)):
	# extract a 3D color histogram from the HSV color space using
	# the supplied number of `bins` per channel
	hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
	hist = cv2.calcHist([hsv], [0, 1, 2], None, bins,
		[0, 180, 0, 256, 0, 256])
	# handle normalizing the histogram if we are using OpenCV 2.4.X
	if imutils.is_cv2():
		hist = cv2.normalize(hist)
	# otherwise, perform "in place" normalization in OpenCV 3 (I
	# personally hate the way this is done
	else:
		cv2.normalize(hist, hist)
	# return the flattened histogram as the feature vector
	return hist.flatten()


# In[4]:


train_args = {
    'dataset': './train',
    'neighbors': 10,
    'jobs': -1
}

val_args = {
    'dataset': './val',
    'neighbors': 10,
    'jobs': -1
}

test_args = {
    'dataset': './test',
    'neighbors': 10,
    'jobs': -1
}


# In[5]:


# grab the list of images that we'll be describing
print("[INFO] describing images...")
train_imagePaths = list(paths.list_images(train_args["dataset"]))
val_imagePaths = list(paths.list_images(val_args["dataset"]))
test_imagePaths = list(paths.list_images(test_args["dataset"]))

# initialize the raw pixel intensities matrix, the features matrix,
# and labels list
train_rawImages = []
train_features = []
train_labels = []
val_rawImages = []
val_features = []
val_labels = []
test_rawImages = []
test_features = []
test_labels = []

#print(train_imagePaths)


# In[6]:


# Randomizes list and pulls a subset for faster debugging

import random

#random.shuffle(train_imagePaths)
#train_imagePaths = train_imagePaths[:50]


# In[7]:


# loop over the input images
for (i, imagePath) in enumerate(train_imagePaths):
	# load the image and extract the class label (assuming that our
	# path as the format: /path/to/dataset/{class}.{image_num}.jpg
	image = cv2.imread(imagePath)
	label = imagePath.split(os.path.sep)[-2]
	# extract raw pixel intensity "features", followed by a color
	# histogram to characterize the color distribution of the pixels
	# in the image
	pixels = image_to_feature_vector(image)
	hist = extract_color_histogram(image)
	# update the raw images, features, and labels matricies,
	# respectively
	train_rawImages.append(pixels)
	train_features.append(hist)
	train_labels.append(label)
	# show an update every 1,000 images
	if i > 0 and i % 1000 == 0:
		print("[INFO] processed {}/{}".format(i, len(train_imagePaths)))
        
# loop over the input images
for (i, imagePath) in enumerate(val_imagePaths):
	# load the image and extract the class label (assuming that our
	# path as the format: /path/to/dataset/{class}.{image_num}.jpg
	image = cv2.imread(imagePath)
	label = imagePath.split(os.path.sep)[-2]
	# extract raw pixel intensity "features", followed by a color
	# histogram to characterize the color distribution of the pixels
	# in the image
	pixels = image_to_feature_vector(image)
	hist = extract_color_histogram(image)
	# update the raw images, features, and labels matricies,
	# respectively
	val_rawImages.append(pixels)
	val_features.append(hist)
	val_labels.append(label)
	# show an update every 1,000 images
	if i > 0 and i % 1000 == 0:
		print("[INFO] processed {}/{}".format(i, len(val_imagePaths)))
        
# loop over the input images
for (i, imagePath) in enumerate(test_imagePaths):
	# load the image and extract the class label (assuming that our
	# path as the format: /path/to/dataset/{class}.{image_num}.jpg
	image = cv2.imread(imagePath)
	label = imagePath.split(os.path.sep)[-2]
	# extract raw pixel intensity "features", followed by a color
	# histogram to characterize the color distribution of the pixels
	# in the image
	pixels = image_to_feature_vector(image)
	hist = extract_color_histogram(image)
	# update the raw images, features, and labels matricies,
	# respectively
	test_rawImages.append(pixels)
	test_features.append(hist)
	test_labels.append(label)
	# show an update every 1,000 images
	if i > 0 and i % 1000 == 0:
		print("[INFO] processed {}/{}".format(i, len(test_imagePaths)))


# In[8]:


# show some information on the memory consumed by the raw images
# matrix and features matrix
train_rawImages = np.array(train_rawImages)
train_features = np.array(train_features)
train_labels = np.array(train_labels)
print("[INFO] pixels matrix: {:.2f}MB".format(
	train_rawImages.nbytes / (1024 * 1000.0)))
print("[INFO] features matrix: {:.2f}MB".format(
	train_features.nbytes / (1024 * 1000.0)))

val_rawImages = np.array(val_rawImages)
val_features = np.array(val_features)
val_labels = np.array(val_labels)
print("[INFO] pixels matrix: {:.2f}MB".format(
	val_rawImages.nbytes / (1024 * 1000.0)))
print("[INFO] features matrix: {:.2f}MB".format(
	val_features.nbytes / (1024 * 1000.0)))

test_rawImages = np.array(test_rawImages)
test_features = np.array(test_features)
test_labels = np.array(test_labels)
print("[INFO] pixels matrix: {:.2f}MB".format(
	test_rawImages.nbytes / (1024 * 1000.0)))
print("[INFO] features matrix: {:.2f}MB".format(
	test_features.nbytes / (1024 * 1000.0)))


# In[20]:


# partition the data into training and testing splits, using 75%
# of the data for training and the remaining 25% for testing
(trainRI, valRI, trainRL, valRL) = train_test_split(
	train_rawImages, train_labels, test_size=0.10, random_state=42)
(trainFeat, valFeat, trainLabels, valLabels) = train_test_split(
	train_features, train_labels, test_size=0.10, random_state=42)

# trainRI = train_rawImages
# trainRL = train_labels
# trainFeat = train_features
# trainLabels = train_labels

# valRI = val_rawImages
# valRL = val_labels
# valFeat = val_features
# valLabels = val_labels

testRI = test_rawImages
testRL = test_labels
testFeat = test_features
testLabels = test_labels


# In[23]:


# Find good k value using val dataset
#train and evaluate a k-NN classifer on the raw pixel intensities
#print("[INFO] evaluating raw pixel accuracy...")
k_acc = [0]
for k in range(1,51):
    model = KNeighborsClassifier(n_neighbors=k,
        n_jobs=train_args["jobs"])
    model.fit(trainRI, trainRL)
    acc = model.score(valRI, valRL)
    k_acc.append(acc)
    print("[INFO] raw pixel accuracy: {:.2f}%".format(acc * 100))
    
k_opt = k_acc.index(max(k_acc))
print(k_opt)


# In[24]:


#train and evaluate a k-NN classifer on the raw pixel intensities
print("[INFO] evaluating raw pixel accuracy...")
model = KNeighborsClassifier(n_neighbors=k_opt,
	n_jobs=train_args["jobs"])
model.fit(trainRI, trainRL)
acc = model.score(testRI, testRL)
print("[INFO] raw pixel accuracy: {:.2f}%".format(acc * 100))


# In[25]:


# Find good k value using val dataset
#train and evaluate a k-NN classifer on the raw pixel intensities
#print("[INFO] evaluating raw pixel accuracy...")
k_acc = [0]
for k in range(1,51):
    model = KNeighborsClassifier(n_neighbors=k,
        n_jobs=train_args["jobs"])
    model.fit(trainFeat, trainLabels)
    acc = model.score(valFeat, valLabels)
    k_acc.append(acc)
    print("[INFO] raw pixel accuracy: {:.2f}%".format(acc * 100))
    
k_opt = k_acc.index(max(k_acc))
print(k_opt)


# In[26]:


# train and evaluate a k-NN classifer on the histogram
# representations
print("[INFO] evaluating histogram accuracy...")
model = KNeighborsClassifier(n_neighbors=k_opt,
	n_jobs=train_args["jobs"])
model.fit(trainFeat, trainLabels)
acc = model.score(testFeat, testLabels)
print("[INFO] histogram accuracy: {:.2f}%".format(acc * 100))


# In[ ]:




