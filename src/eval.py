import sys
sys.path.append('/home/g/git/nodules/models/')
sys.path.append('/home/g/git/nodules/keras/')

import fpr_xrv2_1c3 as m
import generatorModel as gen
import kerasUtil as ku
#from sklearn.metrics import roc_curve, f1_score, auc
from glob import glob
import os
import scipy.misc
import numpy as np
#import matplotlib.pyplot as plt

FILE = '/home/g/git/nodules/eval.dat'
PATH = '/home/g/nodules/trfpr/val/full/1/3/'
WEIGHT = '/home/g/nodules/fprmodels/bin1/3022/78.h5'


def loadpng1(cfg, full=True):
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
      #print label
      #label['malignancy'] = meta[1].split('-')
      #label['vars'] = meta[2].split('-')
      #meta = meta[2].split('-')
      #print meta
      #meta = meta[7].split('.')
      #label['count'] = meta[3]
      #label['slice'] = meta[4]
      labels.append(label)
  if path[-4:] == '.png':
    image = scipy.misc.imread(path)
    image = ku.formatImage(image,full)
    images.append(np.array(image))
    label = path.split('/')
    label = label[-1]
    if label[1] == '.':
      label = label[2:]
    elif label[2] == '.':
      label = label[3:]
    label = str(label)
    meta = label.split('_')
    label = {'labels':meta[0]}
    labels.append(label)
  return images, labels


def eval(FILE=FILE,PATH=PATH,WEIGHT=WEIGHT):
  images, labels = loadpng1(PATH, True)
  images, labels = ku.cleanData(images, labels, 2)
  images = ku.to3d(images, 2, True)
  labels = ku.get2Labels(labels)
  images = ku.convertArray(images)
  images-=0.2
  images/=0.2
  model = m.getModel(2,sz=32,full=True)
  model.compile(loss='categorical_crossentropy',optimizer='adadelta',metrics=['accuracy'])
  fin = open(FILE,'r')
  #fout = open(FILEOUT, 'w')
  for line in fin:
    #model = m.getModel(2,sz=32,full=True)
    #model.compile(loss='categorical_crossentropy',optimizer='adadelta',metrics=['accuracy'])
    line = line.strip()
    line = line.split(',')
    print 'CASE: '+str(line[0])+', EPOCH: '+str(line[1])
    print WEIGHT
    model.load_weights(WEIGHT+str(line[0])+'/'+str(line[1])+'.h5')
    preds = model.predict(images, batch_size=64)
    tmp = np.zeros(preds.shape[0])
    tmp[preds[:,0] > preds[:,1]] = 1
    pos = np.sum(tmp)
    print "False positives: "+str(pos)
    perscan = pos/89. 
    print "False positives per scan: "+str(perscan)
    tlbl = np.zeros(preds.shape[0])
    tlbl[labels[:,0]==1] = 1
    tp_fn = np.sum(tlbl)
    tmp2 = np.zeros(preds.shape[0])
    tmp2[preds[:,0] > preds[:,1]] = 1
    tmp2[tlbl==1] += 1
    tmp3 = np.where(tmp2==2,1,0)
    tp = np.sum(tmp3)
    print "True positives: "+str(tp)
    sensitivity = tp/tp_fn
    print "Sensitivity: "+str(sensitivity)
    tlbl = np.zeros(preds.shape[0])
    tlbl[labels[:,1]==1] = 1
    tn_fp = np.sum(tlbl)
    tmp = np.zeros(preds.shape[0])
    tmp[preds[:,1] > preds[:,0]] = 1
    tn = np.sum(tmp)
    specificity = tn/tn_fp
    print "Specificity: "+str(specificity)
  fin.close()
  #fout.close()
  return 0


if __name__ == '__main__':
  #try:
    test = sys.argv[1]
    val = sys.argv[2]
    #case = sys.argv[3]
    #epoch = sys.argv[4]
    FILE = '/home/g/git/nodules/eval.dat'
    PATH = '/home/g/nodules/trfpr/val/full/'+str(test)+'/'+str(val)+'/'
    try:
      if sys.argv[3]:
        WEIGHT = '/home/g/nodules/fprmodels/bin'+str(test)+'b/'
    except:
      WEIGHT = '/home/g/nodules/fprmodels/bin'+str(test)+'/'
    eval(FILE,PATH,WEIGHT)
  #except:
  #  print "NO COMMAND LINE ARGS..!!"
  #  eval()

