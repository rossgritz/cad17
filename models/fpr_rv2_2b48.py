##
# New model architecture
#
##

from functools import partial

from keras import backend as K
from keras.models import Model, Sequential
from keras.optimizers import Adam
from keras.layers import Input, Convolution3D, MaxPooling3D, UpSampling3D
from keras.layers import Activation, Dense, Flatten, Dropout
from keras.layers.convolutional import Cropping3D
from keras.layers.merge import concatenate, add
from keras.layers.advanced_activations import LeakyReLU

import tensorflow as tf


def printGraph(model):
  from keras.utils import plot_model
  plot_model(model, to_file='currentModel.png')
  

def selu(x):
  alpha = 1.673263242354377
  scale = 1.050700987355480
  return scale*tf.where(x>=0.0, x, alpha*K.exp(x)-alpha)


conv3d = partial(Convolution3D,padding='same',use_bias=True, 
                 kernel_initializer='glorot_normal')


def resiconv(inl, sz=32):
  c1 = conv3d(sz,kernel_size=1)(inl)
  c1 = LeakyReLU(alpha=0.3)(c1)
  c2 = conv3d(sz,kernel_size=1)(inl)
  c2 = LeakyReLU(alpha=0.3)(c2)
  c2b = conv3d(sz,kernel_size=3)(c2)
  c2b = LeakyReLU(alpha=0.3)(c2b)
  c3 = conv3d(sz,kernel_size=1)(inl)
  c3 = LeakyReLU(alpha=0.3)(c3)
  c3b = conv3d(sz,kernel_size=3)(c3)
  c3b = LeakyReLU(alpha=0.3)(c3b)
  c3c = conv3d(sz,kernel_size=3)(c3b)
  c3c = LeakyReLU(alpha=0.3)(c3c)
  concat = concatenate([c1,c2b,c3c],axis=1)
  c4 = conv3d(int(inl.shape[1]),kernel_size=1)(concat)
  c4 = LeakyReLU(alpha=0.3)(c4)
  suml = add([inl,c4])
  relu = LeakyReLU(alpha=0.0)(suml)
  return relu


def spred(inl,var=(16,4,5,7)):
  d,n1,n2,n3 = var[0],var[1],var[2],var[3]
  ishape = int(inl.shape[1])
  p1 = MaxPooling3D(pool_size=(2,2,2))(inl)
  c1 = conv3d(ishape//d*n1,kernel_size=3,strides=2)(inl)
  c1 = LeakyReLU(alpha=0.3)(c1)
  c2 = conv3d(ishape//d*n1,kernel_size=1)(inl)
  c2 = LeakyReLU(alpha=0.3)(c2)
  c2b = conv3d(ishape//d*n2,kernel_size=3,strides=2)(c2)
  c2b = LeakyReLU(alpha=0.3)(c2b)
  c3 = conv3d(ishape//d*n1,kernel_size=1)(inl)
  c3 = LeakyReLU(alpha=0.3)(c3)
  c3b = conv3d(ishape//d*n2,kernel_size=2)(c3)
  c3b = LeakyReLU(alpha=0.3)(c3b)
  c3c = conv3d(ishape//d*n3,kernel_size=3,strides=2)(c3b)
  c3c = LeakyReLU(alpha=0.3)(c3c)
  concat = concatenate([p1,c1,c2b,c3c],axis=1)
  return concat 
  

def getModel(outputs=2, sz=48):
  inputdata = Input((1,sz,sz,sz))  
  c1 = conv3d(sz,kernel_size=3)(inputdata)
  c1 = LeakyReLU(alpha=0.3)(c1)
  sred1 = spred(c1,(16,4,5,7))
  cres1 = resiconv(sred1,sz=24) 
  c2 = conv3d(int(cres1.shape[1])//2,kernel_size=1)(cres1)
  c2 = LeakyReLU(alpha=0.3)(c2)
  cres2 = resiconv(c2,sz=24)
  sred2 = spred(cres2,(16,4,5,7))
  cres3 = resiconv(sred2,sz=24)
  c3 = conv3d(int(cres3.shape[1])//2,kernel_size=1)(cres3)
  c3 = LeakyReLU(alpha=0.3)(c3)
  cres4 = resiconv(c3,sz=24)
  c4 = conv3d(int(cres4.shape[1])//2,kernel_size=1)(cres4)
  c4 = LeakyReLU(alpha=0.3)(c4)
  drop1 = Dropout(0.5)(c4) 
  fl1 = Flatten()(drop1)
  fc1 = Dense(128,use_bias=True)(fl1)
  fc1 = LeakyReLU(alpha=0.3)(fc1)
  output = Dense(outputs,kernel_initializer='zeros')(fc1)
  output = Activation('softmax')(output)
  model = Model(inputs=inputdata,outputs=output)
  print model.summary()
  return model
   

if __name__ == '__main__':
  model = getModel()
  printGraph(model)




