
m keras import backend as K
from keras.models import Model, Sequential, load_model
from keras.optimizers import Adam
from keras.layers import Input, Convolution3D, MaxPooling3D, UpSampling3D
from keras.layers import Activation, Dense, Flatten, Dropout
from keras.layers.convolutional import Cropping3D
from keras.layers.merge import concatenate, add
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.noise import AlphaDropout
K.set_image_dim_ordering('th')

from functools import partial
from glob import glob 
import os

conv3d = partial(Convolution3D,padding='same',use_bias=True,
                 activation='selu',
                 kernel_initializer='lecun_normal')

SAMPLES_PATH = '/scr/data/nodules/testseg/'


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


def reda(inl):
  ishape = int(inl.shape[1])
  p1 = MaxPooling3D(pool_size=(2,2,2))(inl)
  c1 = conv3d(ishape,kernel_size=3,strides=2)(inl)
  c2 = conv3d(ishape,kernel_size=1)(inl)
  c2b = conv3d(ishape,kernel_size=3)(c2)
  c2c = conv3d(ishape,kernel_size=3,strides=2)(c2b)
  concat = concatenate([p1,c1,c2c],axis=1)
  return concat


def redb(inl,var=(16,4,5,7)):
  d,n1,n2,n3 = var[0],var[1],var[2],var[3]
  ishape = int(inl.shape[1])
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


def getModel(outputs=2, sz=64,dropout=0.05):
  inputdata = Input((1,sz,sz,sz))
  c1 = conv3d(sz,kernel_size=3)(inputdata)
  red1 = reda(c1)
  res1 = iresa(red1,sz=32)
  c2 = conv3d(int(res1.shape[1])//2,kernel_size=1)(res1)
  res2 = iresa(c2,sz=32)
  red2 = redb(res2,(16,5,7,8))
  res3 = iresa(red2,sz=32)
  c3 = conv3d(int(res3.shape[1])//2,kernel_size=1)(res3)
  res4 = iresa(c3,sz=32)
  c4 = conv3d(int(res4.shape[1])//2,kernel_size=1)(res4)
  fl1 = Flatten()(c4)
  fc1 = Dense(128,kernel_initializer='lecun_normal')(fl1)
  fc1 = Activation('selu')(fc1)
  drop2 = AlphaDropout(dropout)(fc1)
  output = Dense(outputs,kernel_initializer='zeros')(drop2)#(fc1)
  output = Activation('softmax')(output)
  model = Model(inputs=inputdata,outputs=output)
  print model.summary()
  return model


model = getModel()
model.load_weights('/home/g/Desktop/062317_1305/64-no1305-loss-062317.h5')


for parent, subdir, files in os.walk(SAMPLES_PATH):
  filelist = glob(parent+'/*.png')
