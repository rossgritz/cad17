##
# New model architecture
#
##

from keras import backend as K
from keras.models import Model
from keras.optimizers import Adam
from keras.layers import Input, Convolution3D, MaxPooling3D, UpSampling3D
from keras.layers import Dense, Flatten, Dropout
from keras.layers.convolutional import Cropping3D
from keras.layers.merge import concatenate
from keras.layers.advanced_activations import LeakyReLU


import tensorflow as tf


def printGraph(model):
  from keras.utils import plot_model
  plot_model(model, to_file='model.png')
  

def selu(x):
  alpha = 1.673263242354377
  scale = 1.050700987355480
  return scale*tf.where(x>=0.0, x, alpha*K.exp(x)-alpha)


def getModel():
  inputdata = Input((1,16,32,32))
  conv1a = Convolution3D(128,kernel_size=5,data_format='channels_first',
                        kernel_initializer='glorot_normal',use_bias=True,
                        padding='valid',activation=selu)(inputdata)
  print conv1a.shape
  #conv1a = Activation(selu)(conv1a)  
  dropout1a = Dropout(0.25)(conv1a)
  conv1b = Convolution3D(128,kernel_size=4,data_format='channels_first',
                         kernel_initializer='glorot_normal',use_bias=True,
                         padding='valid',activation=selu)(inputdata)
  print conv1b.shape
  #conv1b = Activation(selu)(conv1b)
  conv2a = Convolution3D(256,kernel_size=4,data_format='channels_first',
                         kernel_initializer='glorot_normal',use_bias=True,
                         padding='valid',activation=selu)(dropout1a)
  print conv2a.shape
  #conv2a = Activation(selu)(conv2a)
  dropout2a = Dropout(0.25)(conv2a)
  pool1b = MaxPooling3D(pool_size=(2,2,2),data_format='channels_first')(conv1b)
  dropout1b = Dropout(0.25)(pool1b)
  print pool1b.shape
  conv2b = Convolution3D(256,kernel_size=3,data_format='channels_first',
                         kernel_initializer='glorot_normal',use_bias=True,
                         padding='valid',activation=selu)(dropout1b)
  print conv2b.shape
  #conv2b = Activation(selu)(conv2b)
  dropout2b = Dropout(0.25)(conv2b)
  conv3a = Convolution3D(256,kernel_size=3,data_format='channels_first',
                         kernel_initializer='glorot_normal',use_bias=True,
                         padding='valid',activation=selu)(dropout2a)
  print conv3a.shape
  #conv3a = Activation(selu)(conv3a)
  pool1a = MaxPooling3D(pool_size=(2,2,2),data_format='channels_first')(conv3a)
  dropout3a = Dropout(0.25)(pool1a)
  print pool1a.shape
  conv3b = Convolution3D(256,kernel_size=2,data_format='channels_first',
                         kernel_initializer='glorot_normal',use_bias=True,
                         padding='valid',activation=selu)(dropout2b)
  print conv3b.shape
  #conv3b = Activation(selu)(conv3b)
  dropout3b = Dropout(0.25)(conv3b)
  concat = concatenate([dropout3a, dropout3b],axis=1)
  print concat.shape
  conv4 = Convolution3D(512,kernel_size=2,data_format='channels_first',
                        kernel_initializer='glorot_normal',use_bias=True,
                        padding='valid',activation=selu)(concat)
  print conv4.shape
  #conv4 = Activation(selu)(conv4)
  pool2 = MaxPooling3D(pool_size=(2,2,2),data_format='channels_first')(conv4)
  print pool2.shape
  dropout4 = Dropout(0.25)(pool2)
  flatten = Flatten()(dropout4)
  print flatten.shape 
  fullyconnected1 = Dense(256,use_bias=True,activation=selu)(flatten)
  #fullyconnected1 = LeakyReLU(alpha=0.1)(fullyconnected1)
  print fullyconnected1.shape
  dropout5 = Dropout(0.25)(fullyconnected1)
  fullyconnected2 = Dense(128,use_bias=True,activation=selu)(dropout5)
  #fullyconnected2 = LeakyReLU(alpha=0.1)(fullyconnected2)
  print fullyconnected2.shape
  dropout6 = Dropout(0.25)(fullyconnected2)
  fullyconnected3 = Dense(64,use_bias=True,activation=selu)(dropout6)
  #fullyconnected3 = LeakyReLU(alpha=0.1)(fullyconnected3)
  print fullyconnected3.shape
  dropout7 = Dropout(0.25)(fullyconnected3)
  classifier = Dense(2,activation='softmax')(dropout7)
  print classifier.shape
  #classifier = Activation('softmax')
  model = Model(inputs=inputdata,outputs=classifier)
  return model



if __name__ == '__main__':
  model = getModel()
  printGraph(model)




