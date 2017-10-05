##
# New model architecture
#
##

from functools import partial

from keras import backend as K
from keras.models import Model, Sequential
from keras.optimizers import Adam
from keras.layers import Input, Convolution3D, MaxPooling3D, UpSampling3D
from keras.layers import Activation, Dense, Flatten, Dropout, AveragePooling3D
from keras.layers.convolutional import Cropping3D
from keras.layers.merge import concatenate, add
from keras.layers.advanced_activations import LeakyReLU
from keras.initializers import RandomNormal
from keras.layers.noise import AlphaDropout

import tensorflow as tf

import math

def printGraph(model):
  from keras.utils import plot_model
  plot_model(model, to_file='currentModel.png')
  

conv3d = partial(Convolution3D,padding='same',use_bias=True,
                 activation='selu',
                 kernel_initializer='lecun_normal')


def iresa(inl, sz=32):
  c1 = conv3d(sz,kernel_size=1)(inl)
  c2 = conv3d(sz,kernel_size=1)(inl)
  c2b = conv3d(sz,kernel_size=3)(c2)
  c3 = conv3d(sz,kernel_size=1)(inl)
  c3b = conv3d(sz,kernel_size=(3,1,1))(c3)
  c3c = conv3d(sz,kernel_size=(1,3,1))(c3b)
  c3d = conv3d(sz,kernel_size=(1,1,3))(c3c)
  c3e = conv3d(sz,kernel_size=3)(c3d)
  concat = concatenate([c1,c2b,c3e],axis=1)
  c4 = conv3d(int(inl.shape[1]),kernel_size=1)(concat)
  suml = add([inl,c4])
  selu = Activation('selu')(suml)
  return selu


def iresb(inl, sz=32):
  c1 = conv3d(sz,kernel_size=1)(inl)
  c2 = conv3d(sz,kernel_size=1)(inl)
  c2b = conv3d(sz,kernel_size=(3,1,1))(c2)
  c2c = conv3d(sz,kernel_size=(1,3,1))(c2b)
  c2d = conv3d(sz,kernel_size=(1,1,3))(c2c)
  c3 = conv3d(sz,kernel_size=1)(inl)
  c3b = conv3d(sz,kernel_size=3)(c3)
  c3c = conv3d(sz,kernel_size=3)(c3b)
  concat = concatenate([c1,c2d,c3c],axis=1)
  c4 = conv3d(int(inl.shape[1]),kernel_size=1)(concat)
  suml = add([inl,c4])
  selu = Activation('selu')(suml)
  return selu


def iresc(inl, sz=32):
  c1 = conv3d(sz,kernel_size=1)(inl)
  c2 = conv3d(sz,kernel_size=1)(inl)
  c2b = conv3d(sz,kernel_size=3)(c2)
  c3 = conv3d(sz,kernel_size=1)(inl)
  c3b = conv3d(sz*2,kernel_size=3)(c3)
  c4 = conv3d(sz,kernel_size=1)(inl)
  c4b = conv3d(sz,kernel_size=3)(c4)
  c4c = conv3d(sz,kernel_size=3)(c4b)
  concat = concatenate([c1,c2b,c3b,c4c],axis=1)
  c4 = conv3d(int(inl.shape[1]),kernel_size=1)(concat)
  suml = add([inl,c4])
  selu = Activation('selu')(suml)
  return selu


def iresd(inl, sz=32):
  c1 = conv3d(sz,kernel_size=1)(inl)
  c2 = conv3d(sz,kernel_size=1)(inl)
  c2b = conv3d(sz,kernel_size=(3,1,1))(c2)
  c2c = conv3d(sz,kernel_size=(1,3,1))(c2b)
  c2d = conv3d(sz,kernel_size=(1,1,3))(c2c)
  c2e = conv3d(sz,kernel_size=(3,1,1))(c2d)
  c2f = conv3d(sz,kernel_size=(1,3,1))(c2e)
  c2g = conv3d(sz,kernel_size=(1,1,3))(c2f)
  concat = concatenate([c1,c2g],axis=1)
  c4 = conv3d(int(inl.shape[1]),kernel_size=1)(concat)
  suml = add([inl,c4])
  selu = Activation('selu')(suml)
  return selu


def irese(inl, sz=32):
  c1 = conv3d(sz+sz/2,kernel_size=1)(inl)
  c2 = conv3d(sz,kernel_size=1)(inl)
  c2b = conv3d(sz,kernel_size=(3,1,1))(c2)
  c2c = conv3d(sz,kernel_size=(1,3,1))(c2)
  c2d = conv3d(sz,kernel_size=(1,1,3))(c2)
  c3 = conv3d(sz,kernel_size=1)(inl)
  c3b = conv3d(sz,kernel_size=(3,1,1))(c3)
  c3c = conv3d(sz,kernel_size=(1,3,1))(c3b)
  c3d = conv3d(sz,kernel_size=(1,1,3))(c3c)
  c3e = conv3d(sz,kernel_size=(3,1,1))(c3d)
  c3f = conv3d(sz,kernel_size=(1,3,1))(c3d)
  c3g = conv3d(sz,kernel_size=(1,1,3))(c3d)
  concat = concatenate([c1,c2b,c2c,c2d,c3e,c3f,c3g],axis=1)
  c4 = conv3d(int(inl.shape[1]),kernel_size=1)(concat)
  suml = add([inl,c4])
  selu = Activation('selu')(suml)
  return selu


def red(inl,var=(16,4,5,7)):
  d,n1,n2,n3 = var[0],var[1],var[2],var[3]
  ishape = int(inl.shape[1])
  p1 = MaxPooling3D(pool_size=(2,2,2))(inl)
  c1 = conv3d(ishape//d*n1,kernel_size=3,strides=2)(inl)
  c2 = conv3d(ishape//d*n1,kernel_size=1)(inl)
  c2b = conv3d(ishape//d*n2,kernel_size=3,strides=2)(c2)
  c3 = conv3d(ishape//d*n1,kernel_size=1)(inl)
  c3b = conv3d(ishape//d*n2,kernel_size=2)(c3)
  c3c = conv3d(ishape//d*n3,kernel_size=3,strides=2)(c3b)
  concat = concatenate([p1,c1,c2b,c3c],axis=1)
  return concat 


def getModel(outputs=2, sz=64,dropout=0.05):
  inputdata = Input((1,sz,sz,sz))  
  c1 = conv3d(sz,kernel_size=3)(inputdata)
  red1 = red(c1,(16,4,5,7))
  res1 = iresa(red1,sz=32) 
  c2 = conv3d(int(res1.shape[1])//2,kernel_size=1)(res1)
  res2 = iresb(c2,sz=32)
  red2 = red(res2,(16,4,5,7))
  res3 = iresc(red2,sz=32)
  c3 = conv3d(int(res3.shape[1])//2,kernel_size=1)(res3)
  res4 = iresd(c3,sz=32)
  red3 = red(res4,(16,8,10,14))
  res5 = irese(red3,sz=64)
  c4 = conv3d(int(res5.shape[1])//2,kernel_size=1)(res5)
  fl1 = Flatten()(c4)
  fc1 = Dense(128,kernel_initializer='lecun_normal')(fl1)
  fc1 = Activation('selu')(fc1)
  drop2 = AlphaDropout(dropout)(fc1)
  output = Dense(outputs,kernel_initializer='zeros')(drop2)#(fc1)
  output = Activation('softmax')(output)
  model = Model(inputs=inputdata,outputs=output)
  print model.summary()
  return model
   

if __name__ == '__main__':
  model = getModel()
  printGraph(model)




