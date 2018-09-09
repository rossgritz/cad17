##
# Candidate generation for LUNA
# Luna used for generating candidate training datasets
##


#import sys
#sys.path.append('/home/g/Desktop/git/nodules/keras/')
#sys.path.append('/home/g/Desktop/git/nodules/tf/')

#import init
#import kerasUtil as ku
#import segutil as util
import candidates as cd
#import nodule

import os
import math
import copy
import time
import scipy
import random
import skimage.feature
import scipy.ndimage
import scipy.misc as scipy_misc
import scipy.ndimage.morphology as morphology
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import SimpleITK as sitk

from glob import glob
from skimage import measure
from skimage import feature

STARTFILE = 0
DATA_PATH = '/scr/data/nodules/luna/test/subset0/'
DATA_PATH = '/home/g/nodules/'
DATA_CANDIDATES_PATH = '/scr/data/nodules/testseg/cands/true/zero/'
DATA_CANDIDATES_PATH = '/home/g/nodules/'
LABELS_PATH = '/home/rrg0013@auburn.edu/git/nodules/resources/annotations.csv'
LABELS_PATH = '/home/g/nodules/'


def loadItk(f, opts):
  itkImage = sitk.ReadImage(f)
  image = sitk.GetArrayFromImage(itkImage)   
  origin = np.array(list(reversed(itkImage.GetOrigin())))
  spacing = np.array(list(reversed(itkImage.GetSpacing())))
  imageName = f.split('/')
  imageName = imageName[-1]
  imageName = imageName[:-4]
  szfactor = spacing / [1,1,1]
  newShape = image.shape * szfactor
  newShape = np.round(newShape)
  szfactor = newShape / image.shape
  image = scipy.ndimage.interpolation.zoom(image, szfactor, mode='nearest')
  return image, itkImage


def getFilelist(path=DATA_PATH):
  filelist = glob(path+'/*.mhd')
  filepathlist = copy.deepcopy(filelist)
  templist = []
  for f in filelist:
    f = f.split('/')
    f = f[-1]
    templist.append(f[:-4])
  return templist, filepathlist
    

def writeCand(roi,uid,outpath,coords,nodule=False):
  c1,c2,c3 = coords
  bb = np.zeros((512,512))
  if roi.shape != (64,64,64): return
  for i in range(8):
    for j in range(8):
      bb[i*64:(i+1)*64,j*64:(j+1)*64] = roi[i*8+j,:,:]
  if nodule:
    fname = 't_'+str(c1)+'_'+str(c2)+'_'+str(c3)+'_'+str(uid)+'.png'
  else:
    fname = 'f_'+str(c1)+'_'+str(c2)+'_'+str(c3)+'_'+str(uid)+'.png'
  if not os.path.exists(outpath): os.makedirs(outpath)
  fname = outpath + '/' + fname
  scipy.misc.toimage(bb[:,:],cmin=-1000,cmax=1000).save(fname)


def generateCandidates(image, labels, coords, shift, com, outpath, seriesuid):
  cands = []
  noLabels = np.max(labels)
  for coord in coords:
    coord = coord-shift
    c3,c2,c1 = coord
    l = labels[c1,c2,c3]
    if l == 0:
      l = labels[c1+1,c2,c3]
      if l == 0:
        l = labels[c1,c2+1,c3]
        if l == 0:
          l = labels[c1,c2,c3+1]
          if l == 0:
            l = labels[c1-1,c2,c3]
            if l == 0:
              l = labels[c1,c2-1,c3]
              if l == 0:
                l = labels[c1,c2,c3-1]
                if l == 0:
                  print "NO CANDIDATE FOUND"
                  coord = coord + shift
                  coord = coord[::-1]
                  c1,c2,c3 = coord
                  roi = image[c1-32:c1+32,c2-32:c2+32,c3-32:c3+32]
                  writeCand(roi,seriesuid,outpath,coord,True)
                  continue
    cands.append(l)
    actualcom = np.round(com[l-1]).astype(int) + (shift[2],shift[1],shift[0])
    c1,c2,c3 = actualcom
    coord = coord + shift
    coord = coord[::-1]
    #dc = np.array(actualcom-coord)
    #dc = np.sqrt(np.sum(np.multiply(dc,dc)))
    roi = image[c1-32:c1+32,c2-32:c2+32,c3-32:c3+32]
    if roi.shape == (64,64,64):
      writeCand(roi,seriesuid,outpath,actualcom,True)
    #if dc >= 3:
    #  c1,c2,c3 = coord
    #  roi = image[c1-32:c1+32,c2-32:c2+32,c3-32:c3+32]
    #  writeCand(roi,seriesuid,outpath,coord,True)    
  for i in range(0,100):
    r = random.randint(1,noLabels)
    while r in cands:
      r = random.randint(1,noLabels)
    actualcom = np.round(com[r-1]).astype(int) + (shift[2],shift[1],shift[0])
    c1,c2,c3 = actualcom
    roi = image[c1-32:c1+32,c2-32:c2+32,c3-32:c3+32]
    if roi.shape == (64,64,64):
      writeCand(roi,seriesuid,outpath,actualcom,False)
  return 0


def getCurrentNodulesCoords(nodules, origin):
  cx = nodules['coordX']
  cx = cx.as_matrix()
  cy = nodules['coordY']
  cy = cy.as_matrix()
  cz = nodules['coordZ']
  cz = cz.as_matrix()
  coords = []
  for i in range(cx.shape[0]):
    tcx = cx[i]
    tcy = cy[i]
    tcz = cz[i]
    center = np.asarray([tcx,tcy,tcz])
    tcoords = center-origin
    tcoords = np.round(tcoords).astype(int)
    tcoords = abs(tcoords)
    print tcoords
    coords.append(tcoords)
  return coords


def run():
  candlist = pd.read_csv(LABELS_PATH)
  filelist, filepathlist = getFilelist(DATA_PATH)
  fout = open(DATA_CANDIDATES_PATH[:-1]+'.status','w')
  for i in range(STARTFILE,len(filelist)):
    print "FILE NUMBER: " + str(i) + " FILENAME: " + str(filelist[i])
    time1 = time.time()
    currentNodules = candlist[candlist['seriesuid'] == filelist[i]] 
    test = currentNodules.as_matrix()
    if test.shape[0] == 0:
      continue
    fpath = filepathlist[i]
    itk = sitk.ReadImage(fpath)
    origin = np.array(itk.GetOrigin())
    image, _ = loadItk(filepathlist[i],None)
    segmentation = cd.segmentLung(image)
    mask = cd.applyMask(image, segmentation)
    segmentedImage = copy.deepcopy(image)
    segmentedImage[mask==0] 
    zmin,zmax,ymin,ymax,xmin,xmax = cd.findROI(segmentedImage)
    roi = cd.crop(segmentedImage)
    t1 = time.time()
    cands = cd.filterHessian(cd.multiscaleHessian(roi))
    t2 = time.time()
    print "SEGMENTED NODULES IN: " + str(t2-t1) + 's'
    #NEW CODE
    #Masking the ROI prior to volume thresholding
    maskROI = mask[zmin:zmax,ymin:ymax,xmin:xmax]
    cands[maskROI==0] = 0
    #END NEW CODE
    t1 = time.time()
    cands = cd.optVolumeThresholding(cands,mn=8,mx=20000)
    t2 = time.time()
    print "THREHOLDING APPLIED TO CANDIDATES: " + str(t2-t1) + 's'
    coords = getCurrentNodulesCoords(currentNodules,origin)
    labels = measure.label(cands,connectivity=1)
    com = scipy.ndimage.measurements.center_of_mass(labels, labels,list(range(1,np.max(labels)+1)))
    generateCandidates(segmentedImage, labels, coords, (xmin,ymin,zmin), com, DATA_CANDIDATES_PATH, filelist[i])
    time2 = time.time()
    print "TOTAL TIME FOR FILE NUMBER " + str(i) + " IS " + str(time2-time1) + "s"
    fout.write('FINISHED FILE NUMBER '+str(i)+' IN '+str(time2-time1)+'s\n')


def getCenteredNodules():
  candlist = pd.read_csv(LABELS_PATH)
  filelist, filepathlist = getFilelist(DATA_PATH)
  fout = open(DATA_CANDIDATES_PATH[:-1]+'.status','w')
  for i in range(STARTFILE,len(filelist)):
    print "FILE NUMBER: " + str(i) + " FILENAME: " + str(filelist[i])
    time1 = time.time()
    currentNodules = candlist[candlist['seriesuid'] == filelist[i]] 
    test = currentNodules.as_matrix()
    if test.shape[0] == 0:
      continue
    fpath = filepathlist[i]
    itk = sitk.ReadImage(fpath)
    origin = np.array(itk.GetOrigin())
    image, _ = loadItk(filepathlist[i],None)
    coords = getCurrentNodulesCoords(currentNodules,origin)
    for coord in coords:
      c1, c2, c3 = coord
      roi = image[c3-32:c3+32,c2-32:c2+32,c1-32:c1+32]
      if roi.shape == (64,64,64):
        writeCand(roi,filelist[i],DATA_CANDIDATES_PATH,coord,True)


if __name__ == '__main__':
  #run()
  getCenteredNodules()




