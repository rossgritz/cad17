##
# Model for 3d segmentation  
#
##

import sys

sys.path.append('/home/rrg0013@auburn.edu/git/keras/keras/preprocessing/')
sys.path.append('/home/rrg0013@auburn.edu/git/nodules/keras/')

import kerasUtil as ku

import scipy.ndimage

import numpy as np

import os
from glob import glob

from keras import backend as K
from keras.models import Model
from keras.optimizers import Adam
from keras.layers import Input, Convolution3D, MaxPooling3D, UpSampling3D, Dropout
from keras.layers.convolutional import Cropping3D
from keras.layers.merge import concatenate
from keras.layers.advanced_activations import PReLU
from keras.callbacks import ModelCheckpoint
#from keras.utils import plot_model

from image3D import ImageDataGenerator

import kerasUtil as ku

SMOOTHING = 1.e-12
LR = 1.e-5
ITERATIONS = 240
PER_EPOCH = 400
LOSSTHRESH = 0.40
DATAPATH = '/scr/data/nodules/testseg/train/six/images/'
OUTPATH = '/scr/data/nodules/testseg/final/six/c/'
VALPATH = '/scr/data/nodules/testseg/final/six/val/images/'
OUTFILE = '/scr/data/nodules/testseg/final/six/c/status.txt'


def to3d(images, isimage=True):
  newImages = []
  if isimage:
    for image in images:
      newImage = np.ones((64,64,64))
      for i in range(8):
        for j in range(8):
          newImage[i*8+j,:,:] = image[i*64:(i+1)*64,j*64:(j+1)*64]
      newImages.append(newImage)
  else:
    for image in images:
      tempImage = np.ones((64,64,64)) ##BUG??
      for i in range(2,6):
      	for j in range(8):
      	  tempImage[(i-2)*8+j,:,:] = image[i*64:(i+1)*64,j*64:(j+1)*64]
      newImage = tempImage[16:48,16:48,16:48]
      newImages.append(newImage)
  return newImages


def loadData(path):
  imagelist = glob(path+'/*image*.png')
  masklist = glob(path+'/*mask*.png')
  images, masks = [], []
  for fname in imagelist:
  	images.append(scipy.ndimage.imread(fname))
  for fname in masklist:
  	masks.append(scipy.ndimage.imread(fname))
  images = to3d(images)
  masks = to3d(masks, False)
  return images, masks


def diceCoef(ytrue, ypred, k=True):
  smoothing = SMOOTHING
  if k:
    ytruef = K.flatten(ytrue)
    ypredf = K.flatten(ypred)
    intersection = K.sum(ytruef*ypredf)
    return 1.-(2.*intersection+smoothing)/(K.sum(ytruef)+K.sum(ypredf)+smoothing)
  else:
    ytruef = ytrue.flatten()
    ypredf = ypred.flatten()
    intersection = np.sum(ytruef*ypredf)
    return (2.*intersection+smoothing)/(np.sum(ytruef)+np.sum(ypredf)+smoothing)


def getAltModel():
  inputdata = Input((1, 64, 64, 64))
  print "input data " + str(inputdata.shape)
  altconv = Convolution3D(128,kernel_size=11,data_format='channels_first',#128
    activation='relu', padding='valid', use_bias=True, 
    kernel_initializer='glorot_normal')(inputdata)
  altconv = PReLU()(altconv) 
  print "altconv " + str(altconv.shape)
  altpool = MaxPooling3D(pool_size=(2,2,2),data_format='channels_first')(altconv)
  print "altpool " + str(altpool.shape)
  altencode = Convolution3D(128,kernel_size=11,data_format='channels_first',#128
    activation='relu', padding='valid', use_bias=True, 
    kernel_initializer='glorot_normal')(altpool)
  altencode = PReLU()(altencode)
  print "altencode " + str(altencode.shape)
  upsampling1 = UpSampling3D(size=(2,2,2),data_format='channels_first')(altencode)
  finalShape = upsampling1.shape
  print "upsampling " + str(finalShape)
  originalShape = altconv.shape#ALTERNATIVE
  cropShape = int(originalShape[2]/2-finalShape[2]/2),int(originalShape[3]/2-\
    finalShape[3]/2),int(originalShape[4]/2-finalShape[4]/2)
  crop1 = Cropping3D(cropping=cropShape,data_format='channels_first')(altconv)
  print "cropped conv " + str(crop1.shape)
  concatenate1 = concatenate([upsampling1, crop1],axis=1)
  print "concatenate " + str(concatenate1.shape)
  dropout1 = Dropout(0.25)(concatenate1)
  expand1 = Convolution3D(256,kernel_size=3,data_format='channels_first',#256
    activation='relu', padding='valid', use_bias=True, 
    kernel_initializer='glorot_normal')(dropout1)
  expand1 = PReLU()(expand1)
  print "expand " + str(expand1.shape)
  altoutputdata = Convolution3D(1,kernel_size=1,data_format='channels_first',#128
    activation='sigmoid', padding='valid', use_bias=True, 
    kernel_initializer='glorot_normal')(expand1)
  print "output " + str(altoutputdata.shape)
  model = Model(inputs=inputdata, outputs=altoutputdata)
  model.compile(optimizer=Adam(lr=1e-5),loss=diceCoef) 
  return model


def getModel():
  inputdata = Input((1, 64, 64, 64))
  conv1 = Convolution3D(128,kernel_size=3,data_format='channels_first',
  	activation='relu', padding='valid', use_bias=True, 
  	kernel_initializer='glorot_normal')(inputdata)
  conv1 = PReLU()(conv1)
  conv2 = Convolution3D(128,kernel_size=3,data_format='channels_first',
  	activation='relu', padding='valid', use_bias=True, 
  	kernel_initializer='glorot_normal')(conv1)
  conv2 = PReLU()(conv2)
  conv3 = Convolution3D(128,kernel_size=3,data_format='channels_first',
  	activation='relu', padding='valid', use_bias=True, 
  	kernel_initializer='glorot_normal')(conv2)
  conv3 = PReLU()(conv3)
  pool1 = MaxPooling3D(pool_size=(2,2,2),data_format='channels_first')(conv3)
  encode1 = Convolution3D(128,kernel_size=3,data_format='channels_first',
  	activation='relu', padding='valid', use_bias=True, 
  	kernel_initializer='glorot_normal')(pool1)
  encode1 = PReLU()(encode1)
  encode2 = Convolution3D(128,kernel_size=3,data_format='channels_first',
  	activation='relu', padding='valid', use_bias=True, 
  	kernel_initializer='glorot_normal')(encode1)
  encode2 = PReLU()(encode2)
  encode3 = Convolution3D(128,kernel_size=3,data_format='channels_first',
  	activation='relu', padding='valid', use_bias=True, 
  	kernel_initializer='glorot_normal')(encode2)
  encode3 = PReLU()(encode3)
  encode4 = Convolution3D(128,kernel_size=3,data_format='channels_first',
  	activation='relu', padding='valid', use_bias=True, 
  	kernel_initializer='glorot_normal')(encode3)
  encode4 = PReLU()(encode4)
  upsampling1 = UpSampling3D(size=(2,2,2),data_format='channels_first')(encode4)
  finalShape = upsampling1.shape
  originalShape = conv3.shape
  cropShape = int(originalShape[2]/2-finalShape[2]/2),int(originalShape[3]/2-\
    finalShape[3]/2),int(originalShape[4]/2-finalShape[4]/2)
  crop1 = Cropping3D(cropping=cropShape,data_format='channels_first')(conv3)
  concatenate1 = concatenate([upsampling1, crop1],axis=1)
  dropout1 = Dropout(0.25)(concatenate1)
  expand1 = Convolution3D(256,kernel_size=3,data_format='channels_first',
  	activation='relu', padding='valid', use_bias=True, 
  	kernel_initializer='glorot_normal')(dropout1)
  expand1 = PReLU()(expand1)
  expand2 = Convolution3D(128,kernel_size=3,data_format='channels_first',
  	activation='relu', padding='valid', use_bias=True, 
  	kernel_initializer='glorot_normal')(expand1)
  expand2 = PReLU()(expand2)
  expand3 = Convolution3D(128,kernel_size=3,data_format='channels_first',
  	activation='relu', padding='valid', use_bias=True, 
  	kernel_initializer='glorot_normal')(expand2)
  expand3= PReLU()(expand3)
  expand4 = Convolution3D(128,kernel_size=3,data_format='channels_first',
  	activation='relu', padding='valid', use_bias=True, 
  	kernel_initializer='glorot_normal')(expand3)
  expand4= PReLU()(expand4)
  expand5 = Convolution3D(128,kernel_size=3,data_format='channels_first',
  	activation='relu', padding='valid', use_bias=True, 
  	kernel_initializer='glorot_normal')(expand4)
  expand5= PReLU()(expand5)
  outputdata = Convolution3D(1,kernel_size=1,data_format='channels_first',
  	activation='sigmoid', padding='valid', use_bias=True, 
  	kernel_initializer='glorot_normal')(expand5)
  model = Model(inputs=inputdata, outputs=outputdata)
  model.compile(optimizer=Adam(lr=LR),loss=diceCoef)
  #model.compile(optimizer='adadelta', loss=diceCoef)
  print model.summary() 
  return model


def run():
  model = getModel()
  print "MODEL GENERATED"
  #plot_model(model, to_file='model.png')
  images, masks = loadData(DATAPATH)
  images = ku.convertArray(images)
  masks = ku.convertArray(masks,False)
  print "IMAGES & MASKS LOADED"
  checkpoint = ModelCheckpoint(OUTPATH+'/init_segmentation_061117.h5')
  model.fit(x=images,y=masks,batch_size=2,epochs=20,shuffle=True,callbacks=[checkpoint])


def runGenerator():
  model = getModel()
  #model.load_weights(OUTPATH+'/lowvalloss.h5')
  print "MODEL GENERATED"
  checkpoint = ModelCheckpoint(OUTPATH+'/lowvalloss.h5',monitor='val_loss',
			       save_weights_only=True,save_best_only=True)
  last = ModelCheckpoint(OUTPATH+'/lowvalloss.h5',monitor='val_loss',
                               save_weights_only=True,save_best_only=False)
  datagen = ImageDataGenerator(
    featurewise_center=False, #center image mean for dataset at zero
    featurewise_std_normalization=False, #divide image by dataset std dev
    zca_whitening=False, #TODO: TRY PCA WHITENING
    rotation_range_x=20., #TODO: AT LEAST ONE MODEL WITH DATA FROM ALL POSSIBLE AXES & ROTATION
    rotation_range_y=20.,
    rotation_range_z=20.,
    x_shift_range=(0.21875),#2./trainImages.shape[4]), #TODO: LOOK AT IMPLEMENTATION TO VERIFY
    y_shift_range=(0.21875),#2./trainImages.shape[3]), #TODO: VERIFY SHIFT IS EQUAL TO CROPPING
    z_shift_range=(0.21875),#2./trainImages.shape[2]),
    x_flip=False,#True,
    y_flip=False,#True,
    z_flip=False,#True,
    fill_mode='constant', #fill region outside boundaries w/ constant
    cval=0, #set at image mean zero
    shear_range=0., #set to zero for no effect
    zoom_range=0., #set to zero for no effect
    data_format='channels_first')
  valgen = ImageDataGenerator(
    featurewise_center=False,
    featurewise_std_normalization=False,
    rotation_range_x=0.,
    rotation_range_y=0.,
    rotation_range_z=0.,
    x_shift_range=(0.0),
    y_shift_range=(0.0),
    z_shift_range=(0.0),
    x_flip=False,
    y_flip=False,
    z_flip=False,
    data_format='channels_first')
  ct, iterations = 0, ITERATIONS
  while ct < iterations:
    metrics = model.fit_generator(datagen.flow_from_segmentation_directory(DATAPATH,batch_size=1,shuffle=True
                        #,save_to_dir=OUTPATH,save_prefix='sample',save_format='png'
                        ),steps_per_epoch=PER_EPOCH,epochs=1,workers=1,callbacks=[checkpoint,last]
                        ,validation_data=valgen.flow_from_segmentation_directory(VALPATH,batch_size=1,shuffle=False)
                        ,validation_steps=40
                        )
    ct += 1
    print "EPOCH #" + str(ct)
    loss = metrics.history['val_loss']
    f = open(OUTFILE,'a+')
    f.write("EPOCH # "+str(ct)+" -- VAL LOSS: " +str(loss[0])+"-- TRAIN LOSS: " + str(metrics.history['loss']) + "\n")
    f.close()
    if float(loss[0]) <= LOSSTHRESH:
      os.rename(OUTPATH+'/lowvalloss.h5',OUTPATH+'/'+str(ct)+'.h5')
    #if loss[0] > 0.96:
    #  print loss
    #  return True
  return False


if __name__ == '__main__':
  #runGenerator()
  getModel()
  #go = True
  #while go:
  #  go = runGenerator()
  #run()



