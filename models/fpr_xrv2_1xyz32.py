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


def getModel(outputs=2, sz=32,dropout=0.05,full=True):
  if full:
    fdim = sz
  else:
    fdim = 16
  inputdata = Input((1,fdim,sz,sz))  
  crop8x = Cropping3D(cropping=(12,0,0),data_format='channels_first')(inputdata)
  crop8y = Cropping3D(cropping=(0,12,0),data_format='channels_first')(inputdata)
  crop8z = Cropping3D(cropping=(12,12,0),data_format='channels_first')(inputdata)

  cz8c1 = conv3d(sz,kernel_size=3)(crop8z)
  cz8red1 = redb(cz8c1,(16,3,5,7))
  cz8res1 = iresa(cz8red1,sz=32,Ishape=124)
  cz8c2 = conv3d(int(cz8res1.shape[1])//2,kernel_size=1)(cz8res1)
  cz8res2 = iresa(cz8c2,sz=32,Ishape=62)
  cz8red2 = redb(cz8res2,(16,5,7,9))
  cz8res3 = iresa(cz8red2,sz=32,Ishape=125)
  cz8c3 = conv3d(int(cz8res3.shape[1])//2,kernel_size=1)(cz8res3)
  cz8res4 = iresa(cz8c3,sz=32,Ishape=62)
  cz8c4 = conv3d(int(cz8res4.shape[1])//2,kernel_size=1)(cz8res4)
  cz8fl1 = Flatten()(cz8c4)
  cz8fl1 = Flatten()(cz8c4)
  cz8fc1 = Dense(128,kernel_initializer='lecun_normal')(cz8fl1)
  cz8fc1 = Activation('selu')(cz8fc1)

  cy8c1 = conv3d(sz,kernel_size=3)(crop8y)
  cy8red1 = redb(cy8c1,(16,3,5,7))
  cy8res1 = iresa(cy8red1,sz=32,Ishape=124)
  cy8c2 = conv3d(int(cy8res1.shape[1])//2,kernel_size=1)(cy8res1)
  cy8res2 = iresa(cy8c2,sz=32,Ishape=62)
  cy8red2 = redb(cy8res2,(16,5,7,9))
  cy8res3 = iresa(cy8red2,sz=32,Ishape=125)
  cy8c3 = conv3d(int(cy8res3.shape[1])//2,kernel_size=1)(cy8res3)
  cy8res4 = iresa(cy8c3,sz=32,Ishape=62)
  cy8c4 = conv3d(int(cy8res4.shape[1])//2,kernel_size=1)(cy8res4)
  cy8fl1 = Flatten()(cy8c4)
  cy8fl1 = Flatten()(cy8c4)
  cy8fc1 = Dense(128,kernel_initializer='lecun_normal')(cy8fl1)
  cy8fc1 = Activation('selu')(cy8fc1)

  cx8c1 = conv3d(sz,kernel_size=3)(crop8x)
  cx8red1 = redb(cx8c1,(16,3,5,7))
  cx8res1 = iresa(cx8red1,sz=32,Ishape=124)
  cx8c2 = conv3d(int(cx8res1.shape[1])//2,kernel_size=1)(cx8res1)
  cx8res2 = iresa(cx8c2,sz=32,Ishape=62)
  cx8red2 = redb(cx8res2,(16,5,7,9))
  cx8res3 = iresa(cx8red2,sz=32,Ishape=125)
  cx8c3 = conv3d(int(cx8res3.shape[1])//2,kernel_size=1)(cx8res3)
  cx8res4 = iresa(cx8c3,sz=32,Ishape=62)
  cx8c4 = conv3d(int(cx8res4.shape[1])//2,kernel_size=1)(cx8res4)
  cx8fl1 = Flatten()(cx8c4)
  cx8fl1 = Flatten()(cx8c4)
  cx8fc1 = Dense(128,kernel_initializer='lecun_normal')(cx8fl1)
  cx8fc1 = Activation('selu')(cx8fc1)

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
  concat1 = concatenate([cx8fc1,cy8fc1,cz8fc1,fc1],axis=1)
  drop2 = AlphaDropout(dropout)(concat1)
  output = Dense(outputs,kernel_initializer='zeros')(drop2)#(fc1)
  output = Activation('softmax')(output)
  model = Model(inputs=inputdata,outputs=output)
  print model.summary()
  return model
   

if __name__ == '__main__':
  model = getModel()
  printGraph(model)




