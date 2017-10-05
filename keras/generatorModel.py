##
# Initial model generator for consequtive models
# Initial model using keras ImageDataGenerator
# ImageDataGenerator was modified for 3d in Keras source
##

import os
import sys
import time

sys.path.append('/home/g/Desktop/git/nodules/src/')
sys.path.append('/home/g/Desktop/git/nodules/keras/')
sys.path.append('/home/g/Desktop/git/keras/keras/preprocessing/')
sys.path.append('/home/g/Desktop/git/nodules/models/')
sys.path.append('/home/rrg0013@auburn.edu/git/nodules/src/')
sys.path.append('/home/rrg0013@auburn.edu/git/nodules/keras/')
sys.path.append('/home/rrg0013@auburn.edu/git/nodules/models/')
sys.path.append('/home/rrg0013@auburn.edu/git/keras/keras/preprocessing/')


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
from image3D import ImageDataGenerator
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import tensorflow as tf

OUTPATH = '/scr/data/nodules/init3dConv/'
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
  if rc=='t'or rc=='T'or rc=='True'or rc=='true':rc=True
  if vallabel=='t'or vallabel=='T'or vallabel=='true'or vallabel=='True':vallabel=True
  else:
    vallabel=False
    rc=False
  if flip=='t'or flip=='T'or flip=='True'or flip=='true':flip=True
  else: flip=False
  if full=='t'or full=='T'or full=='True'or full=='true':full=True
  else: full=False
  weights = config['save_weights_only']
  if weights=='t'or weights=='T'or weights=='True'or weights=='true':weights=True
  sm = config['save_full_model']
  if sm=='t'or sm=='T'or sm=='true'or sm=='True':weights=False
  else: weights=True
  if restart=='t'or restart=='T'or restart=='True'or restart=='true':restart=True
  else: restart=False

  trainImages, trainLabels, valImages, valLabels = data
  if size == 48:
    param = 1
  elif size == 32:
    param = 2
  else:
    param = 0

  m = importlib.import_module(config['model'])
  print "LOADING DATA"
  trainImages, trainLabels = ku.cleanData(trainImages, trainLabels, noutputs)
  trainImages = ku.to3d(trainImages, param, full)
  if noutputs == 2:
    trainLabels = ku.get2Labels(trainLabels)
  elif noutputs == 3:
    trainLabels = ku.get3Labels(trainLabels)
  elif noutputs == -1:
    trainLabels = ku.getNoduleValues(trainLabels)
    trainLabels = trainLabels[:,-5]/5. #Average malignancy = -5
  else:
    print "ERROR: IMPROPER LABELS SPECIFIED"
    return True
  trainImages = ku.convertArray(trainImages)
  valImages, valLabels = ku.cleanData(valImages, valLabels, noutputs)
  valImages = ku.to3d(valImages, param, full)
  if noutputs == 2:
    valLabels = ku.get2Labels(valLabels)
  elif noutputs == 3:
    valLabels = ku.get3Labels(valLabels)
  elif noutputs == -1:
    valLabels = ku.getNoduleValues(valLabels)
    valLabels = valLabels[:,-5]/5.
    noutputs = 1
  if not vallabel:
    try:
      valLabels = np.reshape(valLabels[:,0],valLabels.shape[0])
    except IndexError:
      pass

  valImages = ku.convertArray(valImages)

  std = np.std(trainImages)+1e-12
  mean = np.mean(trainImages)+1e-12
  trainImages -= mean
  trainImages /= std

  print mean
  print std

  datagen = ImageDataGenerator(
    featurewise_center=False,#True, #center image mean for dataset at zero
    featurewise_std_normalization=False, #True, #divide image by dataset std dev
    zca_whitening=False,
    rotation_range_x=rotation, 
    rotation_range_y=rotation,
    rotation_range_z=rotation,
    x_flip=flip,
    y_flip=flip,
    z_flip=flip,
    x_shift_range=shift/trainImages.shape[2],
    y_shift_range=shift/trainImages.shape[3], 
    z_shift_range=shift/trainImages.shape[4],
    fill_mode='constant', #fill region outside boundaries w/ constant
    cval=0, #set at image mean zero
    shear_range=0., #set to zero for no effect
    zoom_range=0., #set to zero for no effect
    data_format='channels_first')
  datagen.fit(trainImages)
  valgen = ImageDataGenerator(
    featurewise_center=False,
    featurewise_std_normalization=False,
    data_format='channels_first')
  valgen.fit(valImages)
  valImages -= mean
  valImages /= std

  if not os.path.exists('/scr/data/nodules/init3dConv/'+date+'/'):
    os.makedirs('/scr/data/nodules/init3dConv/'+date+'/')
  if not os.path.exists(OUTPATH+date+'/'+str(count)):
    os.makedirs(OUTPATH+date+'/'+str(count))

  loss = 1.
  ct = 0
  history = {'acc':[],'val_acc':[],'loss':[],'val_loss':[]}
  model = m.getModel(noutputs,sz=size,dropout=dropout,full=full)
  model.compile(loss=lossfunc,optimizer=optimizer,metrics=['accuracy'])
  lossCheckpoint = ModelCheckpoint(OUTPATH+date+'/'+str(count)+'/'+str(size)+'-no'+str(count)+'-loss-'+date+'.h5', 
                                     monitor='val_loss', save_best_only=True,save_weights_only=weights)
  accuracyCheckpoint = ModelCheckpoint(OUTPATH+date+'/'+str(count)+'/'+str(size)+'-no'+str(count)+'-accuracy-'+date+'.h5', 
                                        monitor='val_acc', save_best_only=True,save_weights_only=weights)
  lastCheckpoint = ModelCheckpoint(OUTPATH+date+'/'+str(count)+'/last.h5',monitor='val_acc',save_best_only=False,save_weights_only=weights)

  if restart:
    model = model.load_weights(OUTPATH+date+'/'+str(count)+'/'+str(size)+'-no'+str(count)+'-loss-'+date+'.h5')

  print "FITTING MODEL ... "

  while ct < iterations:
    if ct==0:
      ftmp = open(OUTPATH+date+'/'+str(count)+'/config.json','w+')
      json.dump(config,ftmp)
      ftmp.close()
    t1 = time.time()
    metrics = model.fit_generator(datagen.flow(trainImages,trainLabels,batch_size=batchSize),
                        #steps_per_epoch=trainImages.shape[0]/batchSize
                        steps_per_epoch=int(epochsz)
                        ,epochs=1,workers=8,
                        #validation_data=valgen.flow(valImages,valLabels,batch_size=batchSize),#verbose=2,
                        #validation_steps=125,callbacks=[lossCheckpoint,accuracyCheckpoint])
                        validation_data=(valImages,valLabels),callbacks=[lossCheckpoint,accuracyCheckpoint,lastCheckpoint])

    acc = metrics.history['val_acc']
    loss = metrics.history['val_loss']
    history['val_loss'].append(float(loss[0]))
    history['acc'].append(metrics.history['acc'])
    history['val_acc'].append(float(acc[0]))
    history['loss'].append(metrics.history['loss'])
    ct += 1
    if float(acc[0]) > 0.92:
      os.rename(OUTPATH+date+'/'+str(count)+'/last.h5',OUTPATH+date+'/'+str(count)+'/'+str(ct)+'.h5')

    #Commented out saving predictions as features
    #TODO: Look further into ensembles from DNN features
    '''if float(acc[0]) >= max(history['val_acc']):
      prob = model.predict_proba(trainImages, batch_size=32)
      classes = model.predict_classes(trainImages, batch_size=32)
      features = np.c_[prob,classes]
      fname = OUTPATH+date+'/'+str(size)+'-no'+str(count)+'-acc-shape-'+str(features.shape[0])+'-'+\
              str(features.shape[1])+'-'+'features-'+date+'.dat'
      fpa = np.memmap(fname,dtype='float32',mode='w+',shape=features.shape)
      fpa[:,:] = features[:,:]
      del fpa'''

    '''if float(loss[0]) <= min(history['val_loss']):
      prob = model.predict_proba(trainImages, batch_size=32)
      classes = model.predict_classes(trainImages, batch_size=32)
      features = np.c_[prob,classes]
      fname = OUTPATH+date+'/'+str(size)+'-no'+str(count)+'-loss-shape-'+str(features.shape[0])+'-'+\
              str(features.shape[1])+'-'+'features-'+date+'.dat'
      fpb = np.memmap(fname,dtype='float32',mode='w+',shape=features.shape)
      fpb[:,:] = features[:,:]
      del fpb'''

    fout = open(OUTPATH+date+'/'+str(count)+'-log.dat','a+')
    t2 = time.time()
    fout.write('iteration '+str(ct)+': acc='+str(acc[0])+' loss='+str(loss[0])+' time='+str(t2-t1)+'s\n')
    fout.close()
    print "\nITERATION " + str(ct)
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

  ''' 
  if weights:
    lowLoss = m.getModel(noutputs)
    lowLoss.compile(loss=lossfunc,optimizer=optimizer,metrics=['accuracy'])
    lowLoss.load_weights(OUTPATH+date+'/'+str(count)+'/'+str(size)+'-no'+str(count)+'-loss-'+date+'.h5')
    predictions = lowLoss.predict(valImages, batch_size=batchSize)
    fpr, tpr, _ = roc_curve(valLabels[:,0],predictions[:,0])
    #f1 = f1_score(valLabels[:,0],predictions[:,0])
    plt.plot(fpr,tpr)
    plt.title('ROC CURVE MIN LOSS')#: F1 = ' + str(f1))
    plt.xlabel('SPECIFICITY')
    plt.ylabel('SENSITIVITY')
    plt.savefig(OUTPATH+date+'/'+str(count)+'/'+str(size)+'-no'+str(count)+'-lossROC-'+date+'.png')
    plt.clf()
    maxAccuracy = m.getModel(noutputs)
    maxAccuracy.compile(loss=lossfunc,optimizer=optimizer,metrics=['accuracy'])
    maxAccuracy.load_weights(OUTPATH+date+'/'+str(count)+'/'+str(size)+'-no'+str(count)+'-loss-'+date+'.h5')
    predictions = maxAccuracy.predict(valImages, batch_size=batchSize)
    fpr, tpr, _ = roc_curve(valLabels[:,0],predictions[:,0])
    #f1 = f1_score(valLabels[:,0],predications[0,:])
    plt.plot(fpr,tpr)
    plt.title('ROC CURVE MAX ACCURACY')#: F1 = ' + str(f1))
    plt.xlabel('SPECIFICITY')
    plt.ylabel('SENSITIVITY')
    plt.savefig(OUTPATH+date+'/'+str(count)+'/'+str(size)+'-no'+str(count)+'-accuracyROC-'+date+'.png')
    plt.clf()
  else:
    lowLoss = load_model(OUTPATH+date+'/'+str(count)+'/'+str(size)+'-no'+str(count)+'-loss-'+date+'.h5')
    predictions = lowLoss.predict(valImages, batch_size=batchSize)
    fpr, tpr, _ = roc_curve(valLabels[:,0],predictions[:,0])
    #f1 = f1_score(valLabels[:,0],predictions[:,0])
    plt.plot(fpr,tpr)
    plt.title('ROC CURVE MIN LOSS')#: F1 = ' + str(f1))
    plt.xlabel('SPECIFICITY')
    plt.ylabel('SENSITIVITY')
    plt.savefig(OUTPATH+date+'/'+str(count)+'/'+str(size)+'-no'+str(count)+'-lossROC-'+date+'.png')
    plt.clf()
    maxAccuracy = load_model(OUTPATH+date+'/'+str(count)+'/'+str(size)+'-no'+str(count)+'-loss-'+date+'.h5')
    predictions = maxAccuracy.predict(valImages, batch_size=batchSize)
    fpr, tpr, _ = roc_curve(valLabels[:,0],predictions[:,0])
    #f1 = f1_score(valLabels[:,0],predications[0,:])
    plt.plot(fpr,tpr)
    plt.title('ROC CURVE MAX ACCURACY')#: F1 = ' + str(f1))
    plt.xlabel('SPECIFICITY')
    plt.ylabel('SENSITIVITY')
    plt.savefig(OUTPATH+date+'/'+str(count)+'/'+str(size)+'-no'+str(count)+'-accuracyROC-'+date+'.png')
    plt.clf()
  '''
  return False


def main():
  config = open('mconfig.json').read()
  config = json.loads(config)
  full = config['full_cube']
  if full=='t'or full=='T'or full=='true'or full=='True':full=True
  else: full=False
  opts = sys.argv
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
  if classifier:
    trainImages, trainLabels = loadpng2(trainpath,full)#opts[2],full)
    valImages, valLabels = loadpng2(valpath,full)#opts[3],full)
  else:
    trainImages, trainLabels = loadpng1(trainpath,full)
    valImages, valLabels = loadpng1(valpath,full)
  end = count + 10
  data = (trainImages,trainLabels,valImages,valLabels)
  optimizer = 'adam'
  while count < end:
    count += 1
    if runCase(data,count,config):#60,optimizer,date,32)
      break


if __name__ == '__main__':
  main()
