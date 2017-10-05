##
# Candidate generation
# 
##

#import sys

#sys.path.append('/home/g/Desktop/git/nodules/keras/')
#sys.path.append('/home/g/Desktop/git/nodules/tf/')

#import init

import time
import math
import scipy
import skimage.feature
import scipy.ndimage
import scipy.misc as scipy_misc
import scipy.ndimage.morphology as morphology
import matplotlib.pyplot as plt
import numpy as np
#import itertools as it

from skimage import measure
from skimage import feature

from mpl_toolkits.mplot3d.art3d import Poly3DCollection


#Called from segmentLung - moved from segutil (deprecated)
def largestLabel(labels, bg=-1):
  vals, counts = np.unique(labels, return_counts=True)
  counts = counts[vals != bg]
  vals = vals[vals != bg]
  if len(counts) > 0:
    return vals[np.argmax(counts)]
  else:
    return None


#Called from notebooks - moved from segutil (deprecated)
def plot3d(image, threshold=-300):
  img = image.transpose(2,1,0)
  verts, faces, norms, vals = measure.marching_cubes_lewiner(img, threshold)
  fig = plt.figure(figsize=(10, 10))
  ax = fig.add_subplot(111, projection='3d')
  mesh = Poly3DCollection(verts[faces], alpha=0.70)
  faceColor = [0.45, 0.45, 0.75]
  mesh.set_facecolor(faceColor)
  ax.add_collection3d(mesh)
  ax.set_xlim(0, img.shape[0])
  ax.set_ylim(0, img.shape[1])
  ax.set_zlim(0, img.shape[2])
  plt.show()


#Unused - moved from segutil (deprecated)
def savePlot3d(image, threshold=-300, fname='plot3d.png'):
  img = image.transpose(2,1,0)
  verts, faces, norms, vals = measure.marching_cubes_lewiner(img, threshold)
  fig = plt.figure(figsize=(10, 10))
  ax = fig.add_subplot(111, projection='3d')
  mesh = Poly3DCollection(verts[faces], alpha=0.70)
  faceColor = [0.45, 0.45, 0.75]
  mesh.set_facecolor(faceColor)
  ax.add_collection3d(mesh)
  ax.set_xlim(0, img.shape[0])
  ax.set_ylim(0, img.shape[1])
  ax.set_zlim(0, img.shape[2])
  plt.savefig(fname)


#Segments the lung
def segmentLung(image, fill=True, reseg=False, error=False):
  print 'SEGMENTING LUNG VOLUME ..'
  binaryImage = np.array(image > -550, dtype=np.int8)+1
  labels = measure.label(binaryImage)
  backgroundLabel = labels[0,0,0]
  if error:
    binaryImage[backgroundLabel == labels] = 1
  else:
    binaryImage[backgroundLabel == labels] = 2
  if fill:
    for i, axialSlice in enumerate(binaryImage):
      axialSlice = axialSlice - 1
      labeling = measure.label(axialSlice, background=0)
      lmax = largestLabel(labeling, bg=0)
      if lmax is not None: 
        binaryImage[i][labeling != lmax] = 1
  binaryImage -= 1
  binaryImage = 1-binaryImage
  labels = measure.label(binaryImage, background=0)
  lmax = largestLabel(labels, bg=0)
  if reseg:
    binaryImage[labels==0] = 0
    binaryImage[labels==lmax] = 0
  elif lmax is not None:
    binaryImage[labels != lmax] = 0
  return binaryImage
'''def segmentLung(image, fill=True):
  print 'SEGMENTING LUNG VOLUME ..'
  binaryImage = np.array(image > -550, dtype=np.int8)+1
  labels = measure.label(binaryImage)
  backgroundLabel = labels[0,0,0]
  binaryImage[backgroundLabel == labels] = 2
  if fill:
    for i, axialSlice in enumerate(binaryImage):
      axialSlice = axialSlice - 1
      labeling = measure.label(axialSlice, background=0)
      lmax = largestLabel(labeling, bg=0)
      if lmax is not None: 
        binaryImage[i][labeling != lmax] = 1
  binaryImage -= 1
  binaryImage = 1-binaryImage
  labels = measure.label(binaryImage, background=0)
  lmax = largestLabel(labels, bg=0)
  if lmax is not None:
    binaryImage[labels != lmax] = 0
  return binaryImage'''


#Checks the lung segmentation
#True if failed --> run reseg lung segmentation
def checkSeg(seg): 
  shape = seg.shape
  x,y,z = shape[0],shape[1],shape[2]
  if seg[0,0,0]==0 and seg[x-1,0,0]==0 and seg[0,y-1,0]==0 and seg[0,0,z-1]==0\
     and seg[x-1,y-1,0]==0 and seg[x-1,0,z-1]==0 and seg[0,y-1,z-1]==0 and seg[x-1,y-1,z-1]==0:
    return False
  else:
    return True


#Compute scales for Guassian filter
def getScales(opts,D1=4.,D2=16.,no=6):
  print "GETTING GAUSSIAN SCALES"
  d0, d1, no = D1, D2, no
  sigma = []
  sigma.append(d0/4.)
  for i in range(1, no):
    val = ((d1/d0)**(1./(no-1.)))**i*sigma[0]
    sigma.append(val)
  return sigma


#Compute hessians for range of Guassian filters
def multiscaleHessian(image, opts=None):
  print "COMPUTING HESSIANS FOR ALL GUASSIANS"
  scales = getScales(opts)
  cands = []
  for Sigma in scales:
    img = np.zeros(image.shape)
    hess = np.zeros((image.ndim,image.ndim)+image.shape)
    sig = np.zeros(image.shape)
    image = scipy.ndimage.filters.gaussian_filter(image,sigma=Sigma,order=0) 
    hess[:,:] = approxHessian(image)
    cands.append(hess)
  return cands


#N-dimensional numerical approximation of Hessian matrix
def approxHessian(L):
  print "COMPUTING APPROX HESSIAN"
  hessian = np.zeros((L.ndim,L.ndim)+L.shape,dtype=L.dtype)
  dL = np.gradient(L)
  for i, d1 in enumerate(dL):
    d2L = np.gradient(d1)
    for j, d2 in enumerate(d2L):
      hessian[i,j] = d2
  return hessian


#Apply segmentation mask to original image for separating lung region
def applyMask(image, segFill):
  print "APPLYING MASK"
  mask = np.ones(segFill.shape)
  for i in range(segFill.shape[0]):
    eroded = morphology.binary_erosion(segFill[i,:,:],np.ones([3,3]))
    dilation = morphology.binary_dilation(eroded,np.ones([5,5]))
    labels = measure.label(dilation)
    regions = measure.regionprops(labels)
    yLabels = []
    for prop in regions:
      bbox = prop.bbox
      if bbox[2]-bbox[0]<475 and bbox[3]-bbox[1]<475 and bbox[0]>40 and bbox[2]<472:
        yLabels.append(prop.label)
    mask[i,:,:] = 0
    for l in yLabels:
      mask[i,:,:] = mask[i,:,:] + np.where(labels==l,1,0)
      mask[i,:,:] = morphology.binary_dilation(mask[i,:,:],np.ones([3,3]))
  return mask


#Filter hessian values corresponding to spherical objects at Guassian scales
#Uses THRESHOLD of 1.25
#TODO: optimize this code - takes up to 2 hours per scan
def filterHessian(hessians, opts=None, THRESHOLD=1.25):
  print "FILTERING HESSIAN VALUES"
  hess = hessians[0]
  scales = getScales(opts)
  cands = np.zeros((hess.shape[2],hess.shape[3],hess.shape[4]))
  candidates = np.zeros((hess.shape[2],hess.shape[3],hess.shape[4]),dtype=np.int32)
  lambdas = np.zeros((hess.shape[2],hess.shape[3],hess.shape[4]))
  ct = 0
  for h in range(len(hessians)):
    t1 = time.time()
    print "HESSIAN NO: " + str(ct)
    ct += 1
    hessian = hessians[h]
    #for i, j, k in it.product(range(hess.shape[2]),range(hess.shape[3]),range(hess.shape[4])):  
    for i in range(hess.shape[2]):
      if i%50 == 0:
        print "  SLICE: " + str(i) + '/' + str(hess.shape[2])
      for j in range(hess.shape[3]):
        for k in range(hess.shape[4]):
          lambdas = np.linalg.eigvals(hessian[:,:,i,j,k])
          l1 = lambdas[0]
          l2 = lambdas[1]
          try:
            l3 = lambdas[2]
          except KeyError:
            l3 = l2
          #Are these already ordered?
          if l1 < 0 and l2 < 0 and l3 < 0:
            l1 = max(abs(l1),abs(l2),abs(l3))
            l2 = min(abs(l1),abs(l2),abs(l3))
          else: continue
          response = (l2*l2)/l1
          if response > candidates[i,j,k] and response > THRESHOLD:
            candidates[i,j,k] = 1
    print "TIME: " + str(time.time()-t1)
  return candidates


#Isolates minimum region of interest circumscribing the lung
#Reduces computational workload for nodule segmentation
def findROI(image):
  sums = 0
  zmin = 0
  zmax = image.shape[0] - 1
  while sums == 0:
    if zmin==zmax:
      return 0,0,0,0,0,0
    #print zmin
    #print zmax
    sumMax = np.sum(image[zmax,:,:])
    sumMin = np.sum(image[zmin,:,:])
    if sumMax == 0:
      zmax -= 1
    if zmin==zmax:
      return 0,0,0,0,0,0
    if sumMin == 0:
      zmin += 1
    sums = sumMax*sumMin
  sums = 0
  xmin = 0
  xmax = image.shape[1] - 1
  while sums == 0:
    sumMax = np.sum(image[:,xmax,:])
    sumMin = np.sum(image[:,xmin,:])
    if sumMax == 0:
      xmax -= 1
    if sumMin == 0:
      xmin += 1
    sums = sumMax*sumMin
  sums = 0
  ymin = 0
  ymax = image.shape[2] - 1
  while sums == 0:
    sumMax = np.sum(image[:,:,ymax])
    sumMin = np.sum(image[:,:,ymin])
    if sumMax == 0:
      ymax -= 1
    if sumMin == 0:
      ymin += 1
    sums = sumMax*sumMin
  return zmin,zmax,xmin,xmax,ymin,ymax


#Crops roi for reducing computational workload in nodule segmentation
def crop(image):
  zmin,zmax,xmin,xmax,ymin,ymax = findROI(image)
  croppedImage = image[zmin:zmax,xmin:xmax,ymin:ymax]
  return croppedImage


#Deprecated
#Thresholding to remove small/large candidates
def volumeThresholding(oldCands, mn=8, mx=20000):
  print "PRECISION VOLUME THRESHOLDING"
  labels = np.zeros(oldCands.shape)
  labels = measure.label(oldCands,connectivity=1)
  cands = np.zeros(oldCands.shape)
  mxlabel = np.max(labels)
  for i in range(1,int(mxlabel)+1):
    count = np.sum(oldCands[labels == i])
    if count > mn and count < mx:
      cands[labels == i] = i
  newCands = np.zeros(cands.shape,dtype=np.int32)
  newCands[cands > 0] = 1
  return newCands


#Optimized volume thresholding for removal of small/large candidates
def optVolumeThresholding(oldCands, mn=8, mx=20000):
  print "OPTIMIZED VOLUME THRESHOLDING"
  mn = mn+1
  mx = mx-1
  labels = measure.label(oldCands,connectivity=1)
  vals, counts = np.unique(labels, return_counts=True)
  hashtable = dict(zip(vals,counts))
  newCands = np.vectorize(hashtable.get)(labels)
  newCands[newCands < mn] = 0
  newCands[newCands > mx] = 0
  newCands[newCands > 0] = 1
  return newCands
