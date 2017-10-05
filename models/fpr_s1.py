##
# New model architecture
#
##

from keras import backend as K
from keras.models import Model, Sequential
from keras.optimizers import Adam
from keras.layers import Input, Convolution3D, MaxPooling3D, UpSampling3D
from keras.layers import Activation, Dense, Flatten, Dropout
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


def getModel(outputs=2):
  model = Sequential()
  model.add(Convolution3D(128,kernel_size=4,use_bias=True,
            input_shape=(1,16,32,32),kernel_initializer='glorot_normal'))
  model.add(LeakyReLU(alpha=0.3))
  model.add(Dropout(0.25))
  model.add(MaxPooling3D(pool_size=(2,2,2)))
  model.add(Convolution3D(256,kernel_size=3,use_bias=True,
            kernel_initializer='glorot_normal'))
  model.add(LeakyReLU(alpha=0.3))
  model.add(Dropout(0.25))
  model.add(Convolution3D(256,kernel_size=3,use_bias=True,
            kernel_initializer='glorot_normal'))
  model.add(LeakyReLU(alpha=0.3))
  model.add(Dropout(0.25))
  #model.add(Convolution3D(512,kernel_size=3,use_bias=True,activation=selu,
  #          kernel_initializer='glorot_normal'))
  #model.add(Dropout(0.25))
  model.add(MaxPooling3D(pool_size=(1,2,2)))
  model.add(Convolution3D(196,kernel_size=2,use_bias=True,activation=selu,
            kernel_initializer='glorot_normal'))
  model.add(Dropout(0.25))
  model.add(Flatten())
  model.add(Dense(128,use_bias=True))
  model.add(LeakyReLU(alpha=0.3))
  #model.add(Activation(selu))
  model.add(Dropout(0.25))
  model.add(Dense(64,use_bias=True))
  model.add(LeakyReLU(alpha=0.3))
  #model.add(Activation(selu))
  model.add(Dropout(0.25))
  model.add(Dense(outputs,use_bias=True))
  model.add(Activation('softmax'))
  print model.summary()
  return model



if __name__ == '__main__':
  model = getModel()
  #printGraph(model)




