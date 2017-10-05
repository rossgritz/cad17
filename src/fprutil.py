##
# Utilities for 3d nodule segmentation
# Generates training images and masks for unet based segmentation training
##

#import segment
import lidc
import candgen as cg

import os

import numpy as np
import pandas as pd
import SimpleITK as sitk
import skimage.measure as measure

import scipy.ndimage 

import matplotlib.pyplot as plt

LUNA_PATH = '/scr/data/nodules/luna/train/subset7/'
LUNA_AUX_PATH = '/scr/data/nodules/luna/CSVFILES/'
LUNA_OUT_PATH = '/scr/data/nodules/testseg/cands/train3/seven/'
XMLDIR = '/home/rrg0013@auburn.edu/xmlc/'
DEBUG = False


def getCoords(currentNodules, origin, spacing):
  cx = currentNodules['coordX']
  cx = cx.as_matrix()
  cy = currentNodules['coordY']
  cy = cy.as_matrix()
  cz = currentNodules['coordZ']
  cz = cz.as_matrix()
  di = currentNodules['diameter_mm']
  di = di.as_matrix()
  coords = []
  diameter = []
  for nn in range(cx.shape[0]):
    tcx = cx[nn]
    tcy = cy[nn]
    tcz = cz[nn]
    d = di[nn]
    volume = 3.14159265359*(d*d*d/8)*4/3.
    center = np.asarray([tcx,tcy,tcz])
    #newCenter = (center-origin)/spacing
    coord = abs(center-origin)
    coord = np.round(coord).astype(int)
    coords.append(coord)
    diameter.append(d)
  return coords, diameter


#Generate a spherical mask for a single nodule
#Can replace this with each radiologists' roi from lidc-idri
def getNoduleMask(shape, center, radius, origin):
  mask = np.zeros(shape)
  radius = np.rint(radius)
  center = np.round(np.array(center-origin)).astype(int)
  sz = np.arange(int(center[0]-radius),int(center[0]+radius+1))
  sy = np.arange(int(center[1]-radius),int(center[1]+radius+1))
  sx = np.arange(int(center[2]-radius),int(center[2]+radius+1))
  sz,sy,sx = np.meshgrid(sz,sy,sx)
  distance = ((center[0]-sz)**2 +(center[1]-sy)**2+(center[2]-sx)**2)
  distanceMatrix = np.ones_like(mask)*np.inf
  distanceMatrix[sx,sy,sz] = distance
  mask[distanceMatrix <= radius**2] = 1
  return mask

#Generates masks for all nodules in image
#Returns a single binary array
def getImageMask(shape, origin, annotations):
  mask = np.zeros(shape)
  for nodule in annotations:
    mask += getNoduleMask(shape,nodule[:3],nodule[3]/2.,origin)
  mask = np.clip(mask, 0, 1)
  return mask


def getImageMaskROI(scan, shape):
  print shape
  mask = np.zeros(shape)
  for nodule in scan.nodules:
    if nodule.valid:
      for coord in nodule.oroi:
        #print coord
        coord = np.round(np.asarray(coord[::-1])).astype(int)
        #print coord
        mask[coord[0],coord[1],coord[2]] += 1
  mask = np.clip(mask, 0, 1)
  for z in range(mask.shape[0]):
    slice = mask[z,:,:]
    slice[slice==0]=-1
    labels = measure.label(slice,connectivity=1)
    labels[labels==1]=0
    labels[labels>0]=1
    mask[z,:,:] = labels
  return mask


def flattenImage(oldImage):
  newImage = np.zeros((512,512))
  for i in range(0,8):
    for j in range(0,8):
      newImage[i*64:(i+1)*64,j*64:(j+1)*64] = oldImage[i*8+j,:,:]
  return newImage


def getROIs(image, mask, origin, scan):#annotations):
  #print "GETTING ROIS"
  irois, mrois = [], []
  #print annotations.shape
  for nodule in scan.clusters: # annotations:
  #print nodule.shape
    #coords = np.round(np.array(abs(nodule[:3]-origin))).astype(int)
    coords = np.round(nodule.centroid).astype(int)
    z = coords[2]
    y = coords[1]
    x = coords[0]
    if image[z-32:z+32,y-32:y+32,x-32:x+32].shape != (64,64,64):
      continue
    mroi = mask[z-32:z+32,y-32:y+32,x-32:x+32]
    if np.sum(mroi[:,:,0]) > 0:
      print '\n\n\nERROR\n\nMASK ERROR\n\nERROR\n\n\n'
      continue
    mroi = flattenImage(mroi)
    mrois.append(mroi)
    iroi = image[z-32:z+32,y-32:y+32,x-32:x+32]
    iroi = flattenImage(iroi)
    irois.append(iroi)
  return irois, mrois


def getMasks():
  count = 0
  noxml = 0
  for parent, subdir, files in os.walk(LUNA_PATH):
    df = pd.read_csv(LUNA_AUX_PATH+'annotations.csv')
    for fname in files:
      if ".mhd" in fname.lower():
        fpath = os.path.join(parent,fname)
        itk = sitk.ReadImage(fpath)
        image = sitk.GetArrayFromImage(itk)
        origin = np.array(list(reversed(itk.GetOrigin())))
        spacing = np.array(list(reversed(itk.GetSpacing())))
        newShape = image.shape * spacing
        newShape = np.round(newShape)
        szfactor = newShape / image.shape
        fname = fname[:-4]
        annotations = df[df['seriesuid']==fname]
        iorigin = np.array(itk.GetOrigin())
        ispacing = np.array(itk.GetSpacing())
        coords = getCoords(annotations,iorigin,ispacing)
        print fname
        print coords
        scan = lidc.xml(XMLDIR,fname,ispacing,iorigin)
        if scan == None:
          print "\nNO XML\n"
          noxml += 1
          print noxml
          print "\n" 
          continue
        scan.annotations = coords
        lidc.cluster(scan)
        scan.combine()
        scan.validateNodules()
        mask = getImageMaskROI(scan,image.shape)
        annotations = annotations[['coordX','coordY','coordZ','diameter_mm']].as_matrix()
        #if annotations.shape[0] == 0: continue
        image = scipy.ndimage.interpolation.zoom(image, szfactor, mode='nearest')
        mask = scipy.ndimage.interpolation.zoom(mask, szfactor, mode='nearest')
        mask[mask>5e-1]=1
        mask[mask<=5e-1]=0
        irois, mrois = getROIs(image, mask, iorigin, scan)#annotations)
        if len(irois) > 0 and len(mrois) > 0:
          #print "SAVING"
          for i in range(len(irois)):
            mroi = mrois[i]
            iroi = irois[i]
            fname = LUNA_OUT_PATH+'/t_'+str(count)+'.png'
            scipy.misc.toimage(iroi[:,:],cmin=-1000,cmax=1000).save(fname)
            #fname = LUNA_OUT_PATH+'/'+str(count)+'_mask.png'
            #scipy.misc.toimage(mroi[:,:],cmin=0,cmax=1).save(fname)
            count += 1
        print "TOTAL: " + str(count)


if __name__ == '__main__':
  getMasks()

