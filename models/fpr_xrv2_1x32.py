##
# New model architecture
#
##

from functools import partial

from keras import backend as K
from keras.models import Model, Sequential
from keras.optimizers import Adam
from keras.layers import Input, Convolution3D, MaxPooling3D, UpSampling3D
from keras.layers import Activation, Dense, Flatten, Dropout, AveragePooling3D, Cropping3D
from keras.layers.convolutional import Cropping3D
from keras.layers.merge import concatenate, add
from keras.layers.advanced_activations import LeakyReLU
from keras.initializers import RandomNormal
from keras.layers.noise import AlphaDropout

import tensorflow as tf
import theano.tensor as T

import math

def printGraph(model):
  from keras.utils import plot_model
  plot_model(model, to_file='currentModel.png')
  

conv3d = partial(Convolution3D,padding='same',use_bias=True,
                 activation='selu',
                 kernel_initializer='lecun_normal')


def iresa(inl, sz=32, Ishape=0):
  c1 = conv3d(sz,kernel_size=1)(inl)
  c2 = conv3d(sz,kernel_size=1)(inl)
  c2b = conv3d(sz,kernel_size=3)(c2)
  c3 = conv3d(sz,kernel_size=1)(inl)
  c3b = conv3d(sz,kernel_size=(3))(c3)
  c3c = conv3d(sz,kernel_size=3)(c3b)
  c4 = conv3d(sz,kernel_size=1)(inl)
  c4b = conv3d(sz,kernel_size=2)(inl)
  concat = concatenate([c1,c2b,c3c,c4b],axis=1)
  try:
    ishape = int(inl.shape[1])
  except TypeError:
    ishape = Ishape
  c4 = conv3d(ishape,kernel_size=1)(concat)
  print inl.shape[1]
  suml = add([inl,c4])
  selu = Activation('selu')(suml)
  return selu


def iresb(inl, sz=32):
  c1 = conv3d(sz,kernel_size=1)(inl)
  c2 = conv3d(sz,kernel_size=1)(inl)
  c2b = conv3d(sz,kernel_size=(2))(c2)
  c2c = conv3d(sz,kernel_size=(3))(c2b)
  c3 = conv3d(sz,kernel_size=1)(inl)
  c3b = conv3d(sz,kernel_size=3)(c3)
  c3c = conv3d(sz,kernel_size=3)(c3b)
  c4 = conv3d(sz,kernel_size=1)(inl)
  c4b = conv3d(sz,kernel_size=2)(c4)
  concat = concatenate([c1,c2c,c3c,c4b],axis=1)
  c5 = conv3d(int(inl.shape[1]),kernel_size=1)(concat)
  suml = add([inl,c5])
  selu = Activation('selu')(suml)
  return selu


def reda(inl):
  try:
    ishape = int(inl.shape[1])
  except TypeError:
    #print inl.shape
    ishape = T.cast(inl.shape[1],'int32')
  p1 = MaxPooling3D(pool_size=(2,2,2))(inl)
  c1 = conv3d(ishape,kernel_size=3,strides=2)(inl)
  c2 = conv3d(ishape,kernel_size=1)(inl)
  c2b = conv3d(ishape,kernel_size=3)(c2)
  c2c = conv3d(ishape,kernel_size=3,strides=2)(c2b)
  concat = concatenate([p1,c1,c2c],axis=1)
  return concat


def redb(inl,var=(16,4,5,7)):
  d,n1,n2,n3 = var[0],var[1],var[2],var[3]
  try:
    ishape = int(inl.shape[1])
    #print ishape
  except TypeError:
    #print inl.shape
    ishape = 64#ishape = T.cast(inl.shape[1],'int32')  
  p1 = MaxPooling3D(pool_size=(2,2,2))(inl)
  c1 = conv3d(ishape//d*n1,kernel_size=1)(inl)
  c1b = conv3d(ishape//d*n1,kernel_size=3,strides=2)(c1)
  c2 = conv3d(ishape//d*n1,kernel_size=1)(inl)
  c2b = conv3d(ishape//d*n2,kernel_size=3,strides=2)(c2)
  c3 = conv3d(ishape//d*n1,kernel_size=1)(inl)
  c3b = conv3d(ishape//d*n2,kernel_size=2)(c3)
  c3c = conv3d(ishape//d*n3,kernel_size=3,strides=2)(c3b)
  concat = concatenate([p1,c1b,c2b,c3c],axis=1)
  return concat 


def getModel(outputs=2, sz=64,dropout=0.05,full=True):
  if full:
    fdim = sz
  else:
    fdim = 16
  inputdata = Input((1,fdim,sz,sz))  
  crop16 = Cropping3D(cropping=(16,32,32),data_format='channels_first')(inputdata)
  crop8 = Cropping3D(cropping=(8,32,32),data_format='channels_first')(inputdata)

  c16c1 = conv3d(sz,kernel_size=3)(inputdata) 
  c16red1 = redb(c16c1,(16,3,5,7))
  c16res1 = iresa(c16red1,sz=32,Ishape=124) 
  c16c2 = conv3d(int(c16res1.shape[1])//2,kernel_size=1)(c16res1)
  c16res2 = iresa(c16c2,sz=32,Ishape=62)
  c16red2 = redb(c16res2,(16,5,7,9))
  c16res3 = iresa(c16red2,sz=32,Ishape=125)
  c16c3 = conv3d(int(c16res3.shape[1])//2,kernel_size=1)(c16res3)
  c16res4 = iresa(c16c3,sz=32,Ishape=62)
  c16c4 = conv3d(int(c16res4.shape[1])//2,kernel_size=1)(c16res4)
  c16fl1 = Flatten()(c16c4)
  c16fl1 = Flatten()(c16c4)
  c16fc1 = Dense(128,kernel_initializer='lecun_normal')(c16fl1)
  c16fc1 = Activation('selu')(c16fc1)

  c1 = conv3d(sz,kernel_size=3)(inputdata) 
  red1 = redb(c1,(16,3,5,7))
  res1 = iresa(red1,sz=32,Ishape=124) 
  c2 = conv3d(int(res1.shape[1])//2,kernel_size=1)(res1)
  res2 = iresa(c2,sz=32,Ishape=62)
  red2 = redb(res2,(16,5,7,9))
  res3 = iresa(red2,sz=32,Ishape=125)
  c3 = conv3d(int(res3.shape[1])//2,kernel_size=1)(res3)
  res4 = iresa(c3,sz=32,Ishape=62)
  c4 = conv3d(int(res4.shape[1])//2,kernel_size=1)(res4)
  fl1 = Flatten()(c4)
  fc1 = Dense(128,kernel_initializer='lecun_normal')(fl1)
  fc1 = Activation('selu')(fc1)
  concat1 = concatenate([c16fc1,fc1],axis=1)
  drop2 = AlphaDropout(dropout)(concat1)
  output = Dense(outputs,kernel_initializer='zeros')(drop2)#(fc1)
  output = Activation('softmax')(output)
  model = Model(inputs=inputdata,outputs=output)
  print model.summary()
  return model
   

if __name__ == '__main__':
  model = getModel()
  printGraph(model)




