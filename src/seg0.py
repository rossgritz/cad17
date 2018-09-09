##
# Model for 3d segmentation  
#
##

import sys
import time
import datetime
sys.path.append('/home/g/git/nodules/keras/')

import kerasUtil as ku
import scipy.ndimage
import numpy as np

import os
import subprocess

from glob import glob
from skimage import measure

from keras import backend as K
from keras.models import Model
from keras.optimizers import Adam
from keras.layers import Input, Convolution3D, MaxPooling3D, UpSampling3D, Dropout, GaussianNoise, BatchNormalization
from keras.layers.convolutional import Cropping3D
from keras.layers.merge import concatenate
from keras.layers.advanced_activations import PReLU
from keras.callbacks import ModelCheckpoint
from image3D import ImageDataGenerator
import tensorflow as tf

import unet2

CASEFLAG = False
def getCaseFromTop():
  global CASEFLAG
  l = []
  fin = open('segcases.csv','r+')
  for line in fin:
     l.append(line)
  l.reverse()
  try:
    x = l.pop()
    CASEFLAG = True
  except:
    return None, None
  l.reverse()
  fin.close()
  x = x.strip()
  x = x.split(',')
  test, val = x[0], x[1]
  #TODO: OUTPUT CASE RECORD CSV w/ GPU
  fout = open('tmp.csv','w+')
  for v in l:
    fout.write(v)
  fout.close()
  subprocess.call('mv tmp.csv segcases.csv', shell=True)
  return (test, val)


LOSSEARLY = [0.0,0.38,0.52,0.42,0.37,0.47,0.41,0.4,0.34,0.29]
LOSSLATE = [0.0,0.35,0.49,0.39,0.34,0.44,0.38,0.37,0.31,0.25]
VAL_COUNTS=[0,31,14,12,32,19,27,25,18,16]
CASE = getCaseFromTop()
test, val = CASE
if test == None:
  TEST = 0
  VAL = 0
else:
  TEST = test
  VAL = val
GPU = 0
BATCH = 2
OUTDIR = 0
FLIP = False#True
ANGLE = 20.0
SMOOTHING = 1.e-12
LR = 1.0e-5
LOWLOSS = 0.6
ITERATIONS = 210
PER_EPOCH = 215
VAL_COUNT = VAL_COUNTS[int(VAL)]#55
LOSSTHRESH = 0.6
RESTARTa = 0.85
RESTARTb = 0.65
OUTDIR = 0
FPCHECK = 1
DATAPATH = '/home/g/nodules/trseg/'+str(TEST)+'/'+str(VAL)+'/images/'
OUTPATH = '/home/g/nodules/segmodels/'+str(TEST)+'/'+str(VAL)+'/'+str(OUTDIR)+'/'
VALPATH = '/home/g/nodules/trseg/'+str(TEST)+'/val2/'+str(VAL)+'/images/'
OUTFILE = '/home/g/nodules/segmodels/'+str(TEST)+'/'+str(VAL)+'/'+str(OUTDIR)+'/status.txt'
def makedirs(outdir):
  global OUTPATH
  global OUTFILE
  if not os.path.exists(OUTPATH): os.makedirs(OUTPATH)
  else:
    if len([name for name in os.listdir(OUTPATH)]) > 15:
      outdir+=1
      OUTPATH = '/home/g/nodules/segmodels/'+str(TEST)+'/'+str(VAL)+'/'+str(outdir)+'/'
      OUTFILE = '/home/g/nodules/segmodels/'+str(TEST)+'/'+str(VAL)+'/'+str(outdir)+'/status.txt'
      makedirs(outdir)
makedirs(OUTDIR)
LASTLOSS = 1.0


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


def convertArray(images,center=True,dtype='float32',mask=False):
  i,j,k = images[0].shape
  shape = (len(images),1,i,j,k)
  newImages = np.ones(shape)
  for i in range(len(images)):
    newImages[i,0,:,:,:] = images[i][:,:,:]
  newImages = newImages.astype(dtype)
  if center and not mask:
    newImages /= 256.
  if mask:
    newImages = np.clip(newImages,0,1)
    newImages = np.where(newImages < 1, 0, newImages)
  return newImages


def loadData(path):
  imagelist = glob(path+'/images/*image.png')
  masklist = glob(path+'/masks/*mask.png')
  images, masks = [], []
  for fname in imagelist:
  	images.append(scipy.ndimage.imread(fname))
  for fname in masklist:
  	masks.append(scipy.ndimage.imread(fname))
  images = to3d(images)
  masks = to3d(masks, False)
  images = convertArray(images)
  print images.shape
  masks = convertArray(masks,mask=True)
  return (images, masks)


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


def getModel(stddev):
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
  batchnorm1 = BatchNormalization()(conv3)
  pool1 = MaxPooling3D(pool_size=(2,2,2),data_format='channels_first')(batchnorm1)
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
  batchnorm2 = BatchNormalization()(encode4)
  upsampling1 = UpSampling3D(size=(2,2,2),data_format='channels_first')(batchnorm2)
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
  print model.summary() 
  return model


def stddev(path):
  images = []
  for parent, subdir, files in os.walk(path):
    for file in files:
      image = scipy.misc.imread(os.path.join(parent,file))
      images.append(image)
  imgdat = np.asarray(images)
  stddev = np.std(imgdat)
  return stddev


def restoreList():
  l = []
  fin = open('segcases.csv','r+')
  for line in fin:
     l.append(line)
  l.reverse()
  x = l.append(str(TEST)+','+str(VAL)+'\n')
  l.reverse()
  fin.close()
  #TODO: OUTPUT CASE RECORD CSV w/ GPU
  fout = open('tmp.csv','w+')
  for v in l:
    fout.write(v)
  fout.close()
  subprocess.call('mv tmp.csv segcases.csv', shell=True)


def runGenerator():
  global LASTLOSS
  lowloss = 1.0
  #model = unet3.getModel()
  model = getModel(stddev(DATAPATH))
  #model.load_weights(OUTPATH+'/lowvalloss.h5')
  print "MODEL GENERATED"
  checkpoint = ModelCheckpoint(OUTPATH+'/lowvallossbest.h5',monitor='val_loss',
			       save_weights_only=True,save_best_only=True)
  last = ModelCheckpoint(OUTPATH+'/lowvalloss.h5',monitor='val_loss',
                               save_weights_only=True,save_best_only=False)
  datagen = ImageDataGenerator(
    featurewise_center=True, #center image mean for dataset at zero
    featurewise_std_normalization=True, #divide image by dataset std dev
    zca_whitening=False, #TODO: TRY PCA WHITENING
    rotation_range_x=ANGLE, #TODO: AT LEAST ONE MODEL WITH DATA FROM ALL POSSIBLE AXES & ROTATION
    rotation_range_y=ANGLE,
    rotation_range_z=ANGLE,#0.,#ANGLE,
    x_shift_range=(0.21875),#2./trainImages.shape[4]), #TODO: LOOK AT IMPLEMENTATION TO VERIFY
    y_shift_range=(0.21875),#2./trainImages.shape[3]), #TODO: VERIFY SHIFT IS EQUAL TO CROPPING
    z_shift_range=(0.21875),#2./trainImages.shape[2]),
    x_flip=FLIP,
    y_flip=FLIP,
    z_flip=FLIP,
    fill_mode='constant', #fill region outside boundaries w/ constant
    cval=0, #set at image mean zero
    shear_range=0., #set to zero for no effect
    zoom_range=0., #set to zero for no effect
    data_format='channels_first')
  valgen = ImageDataGenerator(
    featurewise_center=True,
    featurewise_std_normalization=True,
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
  #valdata = loadData(VALPATH)
  ct, iterations = 0, ITERATIONS
  while ct < iterations:
    print "TEST: "+str(TEST)+"   VAL: "+str(VAL)
    metrics = model.fit_generator(datagen.flow_from_segmentation_directory(DATAPATH,batch_size=BATCH,shuffle=True
                        #,save_to_dir=OUTPATH,save_prefix='sample',save_format='png'
                        ),steps_per_epoch=PER_EPOCH,epochs=1,workers=1,callbacks=[checkpoint,last]
                        ,validation_data=valgen.flow_from_segmentation_directory(VALPATH,batch_size=1,shuffle=False)#True)
                        ,validation_steps=VAL_COUNT
                        )
    ct += 1
    print "EPOCH #" + str(ct)
    loss = metrics.history['val_loss']
    f = open(OUTFILE,'a+')
    f.write("EPOCH # "+str(ct)+" -- VAL LOSS, " +str(loss[0])+"-- TRAIN LOSS, " + str(metrics.history['loss']) + "\n")
    f.close()
    if float(loss[0]) <= LOSSTHRESH:
      os.rename(OUTPATH+'/lowvalloss.h5',OUTPATH+'/'+str(ct)+'.h5')
      cmd1 = 'tar -czf '+OUTPATH+'/'+str(ct)+'.tgz -C '+OUTPATH+'/ '+str(ct)+'.h5 '+\
             '&& rm '+OUTPATH+'/'+str(ct)+'.h5'
      subprocess.call([cmd1],shell=True)
    if LASTLOSS == loss[0]:
      return False#True
    else:
      LASTLOSS = loss[0]
    tloss = metrics.history['loss']
    if ct==1 and float(tloss[0]) > RESTARTa:
      print loss
      return False#True
    elif ct==2 and float(tloss[0]) > RESTARTb:
      return False
    if lowloss > loss[0]:
      lowloss = loss[0]
    if ct == 30 and lowloss > LOWLOSS:
      return False#True
    #elif ct == 75 and lowloss > LOSSEARLY[int(VAL)]:
    #  return False
    #elif ct == 120 and lowloss > LOSSLATE[int(VAL)]:
    #  return False
    #FPCHECK=1
    #FPCHECK=1
    nolabels = 0
    #print "FINISHING EPOCH "+str(ct)
    if ct%FPCHECK == 0:
      Y = model.predict_generator(generator=valgen.flow_from_segmentation_directory(VALPATH,batch_size=1,shuffle=False),steps=VAL_COUNT)
      #print Y.shape
      for i in range(Y.shape[0]):
        labels = measure.label(Y[i,0,:,:,:])
        nolabels += np.max(labels)
      print "TOTAL ITEMS IN VALIDATION: " +str(nolabels)
      f = open(OUTFILE,'a+')
      f.write("TOTAL ITEMS IN VALDATA, "+str(nolabels)+" , -1. \n")
      f.close()
      if ct >= 50 and ct <= 60:
        if nolabels >= (int(VAL_COUNT*1.399)+1):
          return False
      #if ct >= 80:
      #  if nolabels >= (int(VAL_COUNT*1.2499)+1):
      #    return False
  ffinish = open('seglog.dat','a+')
  ts = time.time()
  modelno = OUTPATH.strip()
  modelno = modelno.split('/')
  st = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S')
  ffinish.write("TEST: "+str(TEST)+" VAL: "+str(VAL)+" MODEL: "+str(modelno[-2])+' '+st+'\n')
  ffinish.close()
  return True


def main():
  time.sleep(10)
  xx = True
  go = runGenerator()
  if xx:
    if CASEFLAG==True and go==False:
      restoreList()
    command = 'CUDA_VISIBLE_DEVICES='+str(GPU)+' python /home/g/git/nodules/src/seg'+str(GPU)+'.py &'
    subprocess.call([command],shell=True)


if __name__ == '__main__':
  main()

