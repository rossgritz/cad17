import sys
sys.path.append('/home/g/Desktop/git/nodules/src/')
sys.path.append('/home/g/Desktop/git/nodules/keras/')
sys.path.append('/home/g/Desktop/git/nodules/models/')
sys.path.append('/home/g/Desktop/git/keras/keras/preprocessing/')
import os
import copy
from glob import glob
import numpy as np
import pandas as pd
import SimpleITK as sitk
import seg3d 
import candidates as cd
import candgen as cg

datapath = '/scr/data/nodules/luna/train/subset8/'
annotations = '/home/rrg0013@auburn.edu/git/nodules/resources/annotations.csv'

for parent, subdir, files in os.walk(datapath):
  filelist = glob(parent+'/*.mhd')
  filepathlist = copy.deepcopy(filelist)
templist = []
for f in filelist:
  f = f.split('/')
  f = f[-1]
  templist.append(f[:-4])
filelist = templist
del(templist)

no = 4
candlist = pd.read_csv(annotations)
currentNodules = candlist[candlist['seriesuid'] == filelist[no]]
print currentNodules[['coordX','coordY','coordZ','diameter_mm']]

itk = sitk.ReadImage(filepathlist[no])
image, _ = cg.loadItk(filepathlist[no],None)
origin = np.array(itk.GetOrigin())
spacing = np.array(itk.GetSpacing())
cx = currentNodules['coordX']
cx = cx.as_matrix()
cy = currentNodules['coordY']
cy = cy.as_matrix()
cz = currentNodules['coordZ']
cz = cz.as_matrix()
di = currentNodules['diameter_mm']
di = di.as_matrix()

nno = 0
tcx = cx[nno]
tcy = cy[nno]
tcz = cz[nno]
d = di[nno]
volume = 3.14159265359*(d*d*d/8)*4/3.
center = np.asarray([tcx,tcy,tcz])
newCenter = (center-origin)/spacing
coords = center-origin
coords = np.round(coords).astype(int)

def getSegmentationShape(oldshape):
  factors = (np.asarray(oldshape)/32+2)*32
  return factors

ishape = image.shape
segShape = getSegmentationShape(image.shape)
segImage = np.zeros(segShape)
lpad = (segShape-np.asarray(image.shape))/2
hpad = copy.deepcopy(lpad)
if ishape[0]%2>0:
  hpad[0] += 1
if ishape[1]%2>0:
  hpad[1] += 1
if ishape[2]%2>0:
  hpad[2] += 1
segImage[lpad[0]:segShape[0]-hpad[0],lpad[1]:segShape[1]-hpad[1],lpad[2]:segShape[2]-hpad[2]] = image 
stride = 32
strides = copy.deepcopy(segShape)
strides /= stride

##Try strides of 16 and summing ... 
def segmentImage(image, model, strides, stride=32):
  du,dv,dw = strides[0]-1,strides[1]-1,strides[2]-1
  outshape = (du*stride,dv*stride,dw*stride)
  segmentation = np.zeros(outshape)
  for i in range(du):
    for j in range(dv):
      for k in range(dw):
        roi = image[i*32,(i+2)*32,j*32:(j+2)*32,k*32:(k+2)*32]
        segmentation[i*32:(i+1)*32,j*32:(j+1)*32,k*32:(k+1)*32] = model.predict()

model = seg3d.getModel()
model.load_weights('/scr/data/nodules/luna/segtrain/out/lowloss_dsb_weights_060817.h5')
print "\n\n\n\n\nANOTHER SUMMARY\n\n\n\n\n"
#model.compile(optimizer=Adam(lr=1e-5),loss=diceCoef) 
print model.summary()
print model.predict(np.ones((1,1,64,64,64)),batch_size=1)






