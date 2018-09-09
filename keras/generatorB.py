##
# Initial model generator for consequtive models
# Initial model using keras ImageDataGenerator
# ImageDataGenerator was modified for 3d in Keras source
##

import os
import sys
import time
import subprocess

sys.path.append('/home/g/git/nodules/src/')
sys.path.append('/home/g/git/nodules/keras/')
sys.path.append('/home/g/git/keras/keras/preprocessing/')
sys.path.append('/home/g/git/nodules/models/')

import importlib
import kerasUtil as ku
import run
import tensorflow as tf
import numpy as np
import scipy
import json
from keras import backend as K
from keras.models import Sequential 
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution3D, MaxPooling3D
from keras.layers.advanced_activations import LeakyReLU
from keras.models import load_model
from keras.callbacks import ModelCheckpoint
from sklearn.metrics import roc_curve, f1_score
from image3D2 import ImageDataGenerator
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import tensorflow as tf

OUTPATH = '/home/g/nodules/fprmodels/'
ROTATION = 45

#Inputs from filename type 1
def loadpng1(cfg, full):
  path = cfg
  images, labels = [], []
  for parent, subdir, files in os.walk(path):
    for file in files:
      image = scipy.misc.imread(os.path.join(parent,file))
      image = ku.formatImage(image,full)
      images.append(np.array(image))
      label = file
      if label[1] == '.':
        label = label[2:]
      elif label[2] == '.':
        label = label[3:]
      label = str(label)
      meta = label.split('_')
      label = {'labels':meta[0]}
      label['malignancy'] = meta[1].split('-')
      label['vars'] = meta[2].split('-')
      meta = meta[2].split('-')
      meta = meta[7].split('.')
      label['count'] = meta[3]
      label['slice'] = meta[4]
      labels.append(label)
  return images, labels


#Inputs from simpler filename type 2
def loadpng2(cfg, full):
  path = cfg
  images, labels = [], []
  for parent, subdir, files in os.walk(path):
    #print files
    #print "FILES"
    for file in files:
      image = scipy.misc.imread(os.path.join(parent,file))
      image = ku.formatImage(image,full)
      images.append(np.array(image))
      label = file
      label = label.split('_')
      label = label[0]
      label = label.split('/')
      label = label[-1]
      if label == 't':
        label = {'labels':'t'}
      elif label == 'f':
        label = {'labels':'f'}
      labels.append(label)
  return images, labels


def selu(x):
  alpha = 1.673263242354377
  scale = 1.050700987355480
  return scale*tf.where(x>=0.0, x, alpha*K.exp(x)-alpha)


def getModel2():
  return None


def getModel(size):
  model = Sequential()
  model.add(Convolution3D(148,kernel_size=5,strides=(1,2,2),data_format='channels_first',input_shape=(1,16,size,size),kernel_initializer='glorot_normal',use_bias=True))
  #model.add(LeakyReLU(alpha=0.01))
  model.add(Activation(selu))
  model.add(MaxPooling3D(pool_size=(2,2,2)))
  model.add(Dropout(0.5))
  model.add(Convolution3D(256,kernel_size=4,strides=1,data_format='channels_first',kernel_initializer='glorot_normal',use_bias=True))
  #model.add(LeakyReLU(alpha=0.01))
  model.add(Activation(selu))
  model.add(MaxPooling3D(pool_size=(2,2,2)))
  model.add(Dropout(0.5))
  model.add(Flatten())
  model.add(Dense(192,use_bias=True))
  #model.add(LeakyReLU(alpha=0.01))
  model.add(Activation(selu))
  model.add(Dropout(0.5))
  model.add(Dense(128,use_bias=True))
  #model.add(LeakyReLU(alpha=0.01))
  model.add(Activation(selu))
  model.add(Dropout(0.5))
  model.add(Dense(3))
  model.add(Activation('softmax'))
  return model


def focal_loss_func(y_true, y_pred):
  gamma=2.
  alpha=.25
  pt_1 = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))
  pt_0 = tf.where(tf.equal(y_true, 0), y_pred, tf.zeros_like(y_pred))
  return -K.sum(alpha * K.pow(1. - pt_1, gamma) * K.log(pt_1))-K.sum((1-alpha) * K.pow( pt_0, gamma) * K.log(1. - pt_0))


def runCase(data, count, config):# iterations, optimizer, date, size=64):
  iterations = config['iterations']
  optimizer = config['optimizer']
  date = config['date']
  size = config['size']
  rotation = config['rotation']
  dropout = config['dropout']
  batchSize = config['batch_size']
  noutputs = config['outputs']
  full = config['full_cube']
  shift = config['shift']
  flip = config['flip']
  lossfunc = config['loss']
  restart = config['load_weights']
  vallabel = config['categorical_labels']
  rc = config['restart_conditions']
  epochsz = config['epoch']
  thresh1 = config['thresh1']
  thresh1n = config['thresh1n']
  thresh2 = config['thresh2']
  thresh2n = config['thresh2n']
  thresh3 = config['thresh3']
  thresh3n = config['thresh3n']
  focal_loss = config['focal_loss']
  if rc=='t'or rc=='T'or rc=='True'or rc=='true':rc=True
  if vallabel=='t'or vallabel=='T'or vallabel=='true'or vallabel=='True':vallabel=True
  else:
    vallabel=False
    rc=False
  if flip=='t'or flip=='T'or flip=='True'or flip=='true':flip=True
  else: flip=False
  centernorm = config['centernorm']
  if centernorm=='t'or centernorm=='T'or centernorm=='True'or centernorm=='true':centernorm=True
  else: centernorm=False
  if full=='t'or full=='T'or full=='True'or full=='true':full=True
  else: full=False
  weights = config['save_weights_only']
  if weights=='t'or weights=='T'or weights=='True'or weights=='true':weights=True
  sm = config['save_full_model']
  if sm=='t'or sm=='T'or sm=='true'or sm=='True':weights=False
  else: weights=True
  if restart=='t'or restart=='T'or restart=='True'or restart=='true':restart=True
  else: restart=False
  if focal_loss=='t'or focal_loss=='T'or focal_loss=='True'or focal_loss=='true':focal_loss=True
  else:focal_loss=False
  trainpath, valpath = data
  #trainImages, trainLabels, valImages, valLabels = data
  if size == 48:
    param = 1
  elif size == 32:
    param = 2
  else:
    param = 0

  m = importlib.import_module(config['model'])
  print "LOADING DATA"
  #trainImages, trainLabels = ku.cleanData(trainImages, trainLabels, noutputs)
  #trainImages = ku.to3d(trainImages, param, full)
  #if noutputs == 2:
  #  trainLabels = ku.get2Labels(trainLabels)
  #elif noutputs == 3:
  #  trainLabels = ku.get3Labels(trainLabels)
  #elif noutputs == -1:
  #  trainLabels = ku.getNoduleValues(trainLabels)
  #  trainLabels = trainLabels[:,-5]/5. #Average malignancy = -5
  #else:
  #  print "ERROR: IMPROPER LABELS SPECIFIED"
  #  return True
  #trainImages = ku.convertArray(trainImages)
  #valImages, valLabels = ku.cleanData(valImages, valLabels, noutputs)
  #valImages = ku.to3d(valImages, param, full)
  #if noutputs == 2:
  #  valLabels = ku.get2Labels(valLabels)
  #elif noutputs == 3:
  #  valLabels = ku.get3Labels(valLabels)
  #elif noutputs == -1:
  #  valLabels = ku.getNoduleValues(valLabels)
  #  valLabels = valLabels[:,-5]/5.
  #  noutputs = 1
  #if not vallabel:
  #  try:
  #    valLabels = np.reshape(valLabels[:,0],valLabels.shape[0])
  #  except IndexError:
  #    pass

  #valImages = ku.convertArray(valImages)

  #std = np.std(trainImages)+1e-12
  #mean = np.mean(trainImages)+1e-12
  #trainImages -= 0.2#mean
  #trainImages /= 0.2#std

  #print mean
  #print std

  datagen = ImageDataGenerator(
    featurewise_center=centernorm,#False,#True, #center image mean for dataset at zero
    featurewise_std_normalization=centernorm,#False, #True, #divide image by dataset std dev
    zca_whitening=False,
    rotation_range_x=rotation, 
    rotation_range_y=rotation,
    rotation_range_z=rotation,
    x_flip=flip,
    y_flip=flip,
    z_flip=flip,
    x_shift_range=shift/32,#trainImages.shape[2],
    y_shift_range=shift/32,#trainImages.shape[3], 
    z_shift_range=shift/32,#trainImages.shape[4],
    fill_mode='constant', #fill region outside boundaries w/ constant
    cval=0, #set at image mean zero
    shear_range=0., #set to zero for no effect
    zoom_range=0., #set to zero for no effect
    data_format='channels_first')
  #datagen.fit(trainImages)
  valgen = ImageDataGenerator(
    featurewise_center=centernorm,#False,
    featurewise_std_normalization=centernorm,#False,
    data_format='channels_first')
  #valgen.fit(valImages)
  #valImages -= 0.2#mean
  #valImages /= 0.2#std

  if not os.path.exists('/home/g/nodules/fprmodels/'+date+'/'):
    os.makedirs('/home/g/nodules/fprmodels/'+date+'/')
  if not os.path.exists(OUTPATH+date+'/'+str(count)):
    os.makedirs(OUTPATH+date+'/'+str(count))

  loss = 1.
  ct = 0
  history = {'acc':[],'val_acc':[],'loss':[],'val_loss':[]}
  model = m.getModel(noutputs,sz=size,dropout=dropout,full=full)
  if focal_loss:
    model.compile(loss=focal_loss_func,optimizer=optimizer,metrics=['accuracy'])
  else:
    model.compile(loss=lossfunc,optimizer=optimizer,metrics=['accuracy'])
  lossCheckpoint = ModelCheckpoint(OUTPATH+date+'/'+str(count)+'/'+str(size)+'-no'+str(count)+'-loss-'+date+'.h5', 
                                     monitor='val_loss', save_best_only=True,save_weights_only=weights)
  accuracyCheckpoint = ModelCheckpoint(OUTPATH+date+'/'+str(count)+'/'+str(size)+'-no'+str(count)+'-accuracy-'+date+'.h5', 
                                        monitor='val_acc', save_best_only=True,save_weights_only=weights)
  lastCheckpoint = ModelCheckpoint(OUTPATH+date+'/'+str(count)+'/last.h5',monitor='val_acc',save_best_only=False,save_weights_only=weights)

  if restart:
    model = model.load_weights(OUTPATH+date+'/'+str(count)+'/'+str(size)+'-no'+str(count)+'-loss-'+date+'.h5')

  print "FITTING MODEL ... "
  maxAcc = 0.  

  while ct < iterations:
    if ct==0:
      ftmp = open(OUTPATH+date+'/'+str(count)+'/config.json','w+')
      json.dump(config,ftmp)
      ftmp.close()
    t1 = time.time()
    print "ITERATION "+str(ct+1)
    metrics = model.fit_generator(datagen.flow_from_training_directory(trainpath,batch_size=batchSize,shuffle=True),
                        #steps_per_epoch=trainImages.shape[0]/batchSize
                        steps_per_epoch=int(epochsz),verbose=1
                        ,epochs=1,workers=2,
                        validation_data=valgen.flow_from_training_directory(valpath,batch_size=32,shuffle=False),validation_steps=7,#,#verbose=2,
                        #validation_steps=125,callbacks=[lossCheckpoint,accuracyCheckpoint])
                        #validation_data=(valImages,valLabels),
                        callbacks=[lossCheckpoint,accuracyCheckpoint,lastCheckpoint])

    acc = metrics.history['val_acc']
    loss = metrics.history['val_loss']
    history['val_loss'].append(float(loss[0]))
    history['acc'].append(metrics.history['acc'])
    history['val_acc'].append(float(acc[0]))
    history['loss'].append(metrics.history['loss'])
    ct += 1
    if float(acc[0]) > 0.9:
      os.rename(OUTPATH+date+'/'+str(count)+'/last.h5',OUTPATH+date+'/'+str(count)+'/'+str(ct)+'.h5')
    if float(acc[0]) > maxAcc:
      maxAcc = float(acc[0])
    if ct == int(thresh1n) and maxAcc < float(thresh1):
      return True
    elif ct == int(thresh2n)  and maxAcc < float(thresh2):
      return True
    elif ct == int(thresh3n) and maxAcc < float(thresh3):
      return True
    #Commented out saving predictions as features
    #TODO: Look further into ensembles from DNN features

    fout = open(OUTPATH+date+'/'+str(count)+'-log.dat','a+')
    t2 = time.time()
    fout.write('iteration '+str(ct)+': acc='+str(acc[0])+' loss='+str(loss[0])+' time='+str(t2-t1)+'s\n')
    fout.close()
    #print "\nITERATION " + str(ct)
    #if rc:
    #  if acc[0] <= .10 or loss[0] > 2. or loss[0] < 1e-5:
    #    break   
    #if rc:
    #  if acc[0] <= .5:
    #    return

  
  plt.plot(history['acc'])
  plt.plot(history['val_acc'])
  plt.title('MODEL ACCURACY')
  plt.xlabel('EPOCH')
  plt.ylabel('ACCURACY')
  plt.legend(['TRAIN','VALIDATION'],loc='upper left')
  maxAccuracy = round(max(history['val_acc']),3)
  plt.savefig(OUTPATH+date+'/'+str(count)+'/'+str(size)+'-no'+str(count)+'-accuracy'+str(maxAccuracy)+'-'+date+'.png')
  plt.clf()

  plt.plot(history['loss'])
  plt.plot(history['val_loss'])
  plt.title('MODEL LOSS')
  plt.xlabel('EPOCH')
  plt.ylabel('LOSS')
  plt.legend(['TRAIN','VALIDATION'],loc='upper left')
  minLoss = round(min(history['val_loss']),3)
  plt.savefig(OUTPATH+date+'/'+str(count)+'/'+str(size)+'-no'+str(count)+'-loss'+str(minLoss)+'-'+date+'.png')
  plt.clf()

  fout = open(date+'.dat','a+') 
  fout.write(str(count)+','+str(size)+','+str(minLoss)+','+str(maxAccuracy)+','+optimizer+'\n')
  fout.close()

  if noutputs < 2:
    return False 

  return False


def main():
  GPU = sys.argv[2]
  config = open('config'+str(GPU)+'.json').read()
  config = json.loads(config)
  full = config['full_cube']
  if full=='t'or full=='T'or full=='true'or full=='True':full=True
  else: full=False
  opts = sys.argv
  #GPU = sys.argv[2]
  trainpath = config['train']
  valpath = config['val']
  date = config['date']
  mem = float(config['memory_fraction'])
  nooutputs = config['outputs']
  if mem < 0.75:
    #import tensorflow as tf
    from keras.backend.tensorflow_backend import set_session
    configTF = tf.ConfigProto()
    configTF.gpu_options.per_process_gpu_memory_fraction=mem
    set_session(tf.Session(config=configTF))
  count = int(opts[1])
  print count
  #nooutputs = config['outputs']
  classifier = True
  if nooutputs <= 0:
    classifier = False
  #if classifier:
  #  print "TRAIN IMAGES"
  #  trainLabels = loadpng2(trainpath,full)#opts[2],full)
  #  valLabels = loadpng2(valpath,full)#opts[3],full)
  #else:
  #  print "TRAIN IMAGES"
  #  trainImages, trainLabels = loadpng1(trainpath,full)
  #  valImages, valLabels = loadpng1(valpath,full)
  end = count + 10
  data = (trainpath,valpath)
  optimizer = 'adam'
  #while count < end:
  #count += 1
  runCase(data,count,config)#60,optimizer,date,32)
  count += 1
  command = 'CUDA_VISIBLE_DEVICES='+str(GPU)+' python keras/generatorB.py '+str(count)+' '+str(GPU)+' &'
  subprocess.call([command],shell=True)


if __name__ == '__main__':
  main()
