import sys
sys.path.append('/home/rrg0013@auburn.edu/git/nodules/models/')
sys.path.append('/home/rrg0013@auburn.edu/git/nodules/keras/')

import fpr_xrv2_1c as m
import generatorModel as gen
import kerasUtil as ku
from sklearn.metrics import roc_curve, f1_score, auc

import os
import scipy.misc
import numpy as np
import matplotlib.pyplot as plt

PATH = '/scr/data/nodules/testseg/cands/train2/four/'
WEIGHTS = '/scr/data/nodules/init3dConv/072117/4006/64-no4006-accuracy-072117.h5'

PATH = '/scr/data/nodules/testseg/cands/t5/nine/'
#WEIGHTS = '/scr/data/nodules/init3dConv/072117/4006/64-no4006-accuracy-072117.h5'

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
      #print label
      #label['malignancy'] = meta[1].split('-')
      #label['vars'] = meta[2].split('-')
      #meta = meta[2].split('-')
      #print meta
      #meta = meta[7].split('.')
      #label['count'] = meta[3]
      #label['slice'] = meta[4]
      labels.append(label)
  return images, labels

def loadImage():
  images, labels = loadpng1(PATH, True)
  images, labels = ku.cleanData(images, labels, 2)
  images = ku.to3d(images, 0, True)
  labels = ku.get2Labels(labels)
  images = ku.convertArray(images)
  print len(images)
  return images, labels


def eval(images, labels):
  images -= 0.239361703397#0.237466380001#0.239519521595
  images /= 0.221614971758#0.221488147975#0.221684783698

  model = m.getModel(2)
  model.compile(loss='categorical_crossentropy',optimizer='adadelta',metrics=['accuracy'])
  model.load_weights(WEIGHTS)
  preds = model.predict(images, batch_size=8)

  print "STEP 1"

  fpr, tpr, _ = roc_curve(labels[:,0],preds[:,0])
  area = auc(fpr, tpr)
  print "AREA:"
  print area
  plt.plot(fpr,tpr)
  plt.savefig('/home/rrg0013@auburn.edu/four_roc.png')  
  plt.clf()
  tmp = np.zeros(preds.shape[0])
  tmp[preds[:,0] > preds[:,1]] = 1
  pos = np.sum(tmp)
  print pos
  tlbl = np.zeros(preds.shape[0])
  tlbl[labels[:,0]==1] = 1
  tp_fn = np.sum(tlbl)
  print tp_fn
  tmp2 = np.zeros(preds.shape[0])
  tmp2[preds[:,0] > preds[:,1]] = 1
  tmp2[tlbl==1] += 1
  tmp3 = np.where(tmp2==2,1,0)
  tp = np.sum(tmp3)
  print tp
  print "SENSITIVITY: "
  sensitivity = tp/tp_fn
  print sensitivity
  tlbl = np.zeros(preds.shape[0])
  tlbl[labels[:,1]==1] = 1
  tn_fp = np.sum(tlbl)
  #print tn_fp
  tmp = np.zeros(preds.shape[0])
  tmp[preds[:,1] > preds[:,0]] = 1
  tn = np.sum(tmp)
  #print tn
  tmp = np.zeros(preds.shape[0])
  tmp[preds[:,0] > preds[:,1]] = 1
  xpos = np.sum(tmp)
  #print xpos
  perscan = (pos-xpos)/89. 
  print "FP PER SCAN: "
  print perscan 
  tn_fp = tn_fp-(xpos-tp)
  #print tn_fp
  print "SPECIFICITY: "
  specificity = tn/tn_fp
  print specificity


def run():
  images, labels = loadImage()
  std = np.std(images)+1e-12
  mean = np.mean(images)+1e-12
  print mean
  print std


if __name__ == '__main__':
  run()





