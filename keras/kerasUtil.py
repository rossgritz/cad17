import sys
import os

sys.path.append('/home/g/Desktop/git/nodules/src/')
sys.path.append('/home/admin/git/keras/keras/')

import copy
import numpy as np

from keras import backend as K
#import backend as K
from keras.models import Sequential 
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution3D, MaxPooling3D
from keras.layers.advanced_activations import LeakyReLU
from keras.models import load_model


def formatImage(old,full=False):
  if full:
    return old
  old = old[3*64:5*64,:]
  new = copy.deepcopy(old)
  return new


def to3d(images, l=0, full=False):
  newImages = []
  if not full:
    for image in images:
      newImage = np.ones((16,64-l*16,64-l*16))
      for j in range(2):
        for i in range(8):
          newImage[i+j*8,:,:] = image[j*64+8*l:(j+1)*64-8*l,i*64+8*l:(i+1)*64-8*l]
      newImages.append(newImage)
  else:
    for image in images:
      newImage = np.ones((64-l*16,64-l*16,64-l*16))
      for i in range(8-2*l):
        for j in range(8):
          newImage[i*8+j,:,:] = image[(i+l)*64+8*l:(i+l+1)*64-8*l,j*64+8*l:(j+1)*64-8*l]
      newImages.append(newImage)
  return newImages


def get2Labels(labels,test=1):
  classLabels = np.zeros((len(labels)/test,2))
  for i in range(len(labels)/test):
    l = labels[i]
    if l == 'LLL' or l == 'LLLL' or l == 't':# or l == 'LLLS' or l == 'LLLN':
      classLabels[i,0] = 1
    elif l == 'SSS' or l == 'SSSS' or l == 'f':# or l == 'LSSS' or l == 'NSSS':
      classLabels[i,1] = 1
  return classLabels



def get3Labels(labels,test=1):
  classLabels = np.zeros((len(labels)/test,3))
  for i in range(len(labels)/test):
    l = labels[i]
    if l == 'LLL' or l == 'LLLL' or l == 't':# or l == 'LLLS' or l == 'LLLN':
      classLabels[i,0] = 1
    elif l == 'SSS' or l == 'SSSS' or l == 'f':# or l == 'LSSS' or l == 'NSSS':
      classLabels[i,1] = 1
  return classLabels


def getNoduleValues(labels):
  values = np.zeros((len(labels),13))
  for i in range(len(labels)):
    l = labels[i]#i]
    #print l
    varz = l['vars']
    varz[7] = varz[7].split('..')[0]
    label = varz + l['malignancy']
    values[i,:] = np.asarray(label)
    #print values[i,:]
  return values


def convertArray(images,center=True,dtype='float32'):
  i,j,k = images[0].shape
  shape = (len(images),1,i,j,k)
  newImages = np.ones(shape)
  for i in range(len(images)):
    newImages[i,0,:,:,:] = images[i][:,:,:]
  newImages = newImages.astype(dtype)
  if center:
    newImages /= 256.
  return newImages


def cleanData(images,labels,rflag):
  temp = []
  for i in range(len(images)):
    temp.append((images[i],labels[i]))
  images, labels = [], []
  for i in range(len(temp)):
    label = temp[i][1]
    #print label
    l = label['labels']
    if l == 'LLL' or l == 'LLLL' or l == 't':# or l == 'LLLS' or l == 'LLLN':
      images.append(temp[i][0])
      if rflag > 0:
        labels.append(l)
      else:
        labels.append(label)
    elif l == 'SSS' or l == 'SSSS' or l == 'f':# or l == 'LSSS' or l == 'NSSS':
      images.append(temp[i][0])
      if rflag > 0:
        labels.append(l)
      else:
        labels.append(label)
  return images, labels


def getModel():
  model = Sequential()
  model.add(Convolution3D(96,kernel_size=4,strides=(1,1,1),data_format='channels_first',input_shape=(1,16,16,16),kernel_initializer='glorot_normal',use_bias=True))
  model.add(LeakyReLU(alpha=0.01))
  model.add(MaxPooling3D(pool_size=(2,2,2)))
  model.add(Convolution3D(192,kernel_size=3,strides=1,data_format='channels_first',kernel_initializer='glorot_normal',use_bias=True))
  model.add(LeakyReLU(alpha=0.01))
  model.add(MaxPooling3D(pool_size=(2,2,2)))
  model.add(Dropout(0.5))
  model.add(Flatten())
  model.add(Dense(136))
  model.add(LeakyReLU(alpha=0.01))
  model.add(Dropout(0.5))
  model.add(Dense(48))
  model.add(LeakyReLU(alpha=0.01))
  model.add(Dropout(0.5))
  model.add(Dense(2))
  model.add(Activation('softmax'))
  return model
