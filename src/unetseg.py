import SimpleITK as sitk
import numpy as np
import pandas as pd
import scipy
import os
import util
#import sys

from skimage.segmentation import clear_border
from skimage import data
from skimage.morphology import ball, disk, dilation, binary_erosion, remove_small_objects, erosion, closing, reconstruction, binary_closing
from skimage.measure import label,regionprops, perimeter
from skimage.morphology import binary_dilation, binary_opening
from skimage.filters import roberts, sobel
from skimage import measure, feature, morphology
from skimage.transform import resize

from sklearn.cluster import KMeans

from keras.models import Model
from keras.layers import Input, merge, Convolution2D, MaxPooling2D, UpSampling2D, Dropout
from keras.optimizers import Adam, SGD 
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras import backend as K

from tqdm import tqdm
from glob import glob

try:
    from tqdm import tqdm 
except:
    print('TQDM makes much nicer wait bars...')
    tqdm = lambda x: x

LUNA_PATH = '/scr/data/nodules/luna16/train/'
LUNA_AUX_PATH = '/scr/data/nodules/luna16/CSVFILES/'
LUNA_OUT_PATH = '/scr/data/nodules/luna16/luna-work/temp/'



def getFname(flist, case):
  for f in flist:
    if case in f:
      return(f)


def makeMask(center,d,z,w,h,spacing,origin):
  mask = np.zeros([h,w]) 
  vcenter = (center-origin)/spacing
  vd = int(d/spacing[0]+5)
  vxmin = np.max([0,int(vcenter[0]-vd)-5])
  vxmax = np.min([w-1,int(vcenter[0]+vd)+5])
  vymin = np.max([0,int(vcenter[1]-vd)-5]) 
  vymax = np.min([h-1,int(vcenter[1]+vd)+5])
  vxrange = range(vxmin,vxmax+1)
  vyrange = range(vymin,vymax+1)
  for vx in vxrange:
    for vy in vyrange:
      px = spacing[0]*vx + origin[0]
      py = spacing[1]*vy + origin[1]
      if np.linalg.norm(center-np.array([px,py,z]))<=d:
        mask[int((py-origin[1])/spacing[1]),int((px-origin[0])/spacing[0])] = 1.0
  return(mask)


def loadItk(filename):
  itkimage = sitk.ReadImage(filename)
  scan = sitk.GetArrayFromImage(itkimage)   
  origin = np.array(list(reversed(itkimage.GetOrigin())))
  spacing = np.array(list(reversed(itkimage.GetSpacing())))
  return scan, origin, spacing


def world2voxel(worldCoordinates, origin, spacing):
  stretchedVoxelCoordinates = np.absolute(worldCoordinates - origin)
  voxelCoordinates = stretchedVoxelCoordinates / spacing
  return voxelCoordinates


def voxel2world(voxelCoordinates, origin, spacing):
  stretchedVoxelCoordinates = voxelCoordinates * spacing
  worldCoordinates = stretchedVoxelCoordinates + origin
  return worldCoordinates


def seq(start, stop, step=1):
  n = int(round((stop - start)/float(step)))
  if n > 1:
    return([start + step*i for i in range(n+1)])
  else:
    return([])


def drawCircles(image,cands,origin,spacing):
  RESIZE_SPACING = [1, 1, 1]
  imageMask = np.zeros(image.shape)
  imageMasks = []
  for cand in cands:
    radius = np.ceil(ca[4])/2.
    coordX = cand[1]
    coordY = cand[2]
    coordZ = cand[3]
    imageCoords = world2voxel(imageCoord,origin,spacing)
    noduleRange = seq(-radius, radius, RESIZE_SPACING[0])
    for x in noduleRange:
      for y in noduleRange:
        for z in noduleRange:
          coords = world2voxel(np.array((coordZ+z,coordY+y,coordX+x)),origin,spacing)
          if (np.linalg.norm(imageCoords-coords) * RESIZE_SPACING[0]) < radius:
            imageMask[(int)(round(coords[0])),(int)(round(coords[1])),(int)(round(coords[2]))] = int(1)
  return imageMask


def createNoduleMask(imagePath, cands, opts):
  print imagePath
  img, origin, spacing = loadItk(imagePath)
  imageName = imagePath.split('/')
  imageName = imageName[-1]
  imageName = imageName[:-4]
  RESIZE_SPACING = [1, 1, 1]
  resizeFactor = spacing / RESIZE_SPACING
  newRealShape = img.shape * resizeFactor
  newShape = np.round(newRealShape)
  realResize = newShape / img.shape
  newSpacing = spacing / realResize
  lungImg = scipy.ndimage.interpolation.zoom(img, realResize)
  lungMask = util.segmentLung(lungImg)
  try:
    noduleMask = drawCircles(lungImg,cands[imageName],origin,newSpacing)
  except KeyError:
    return
  lungImg512, lungMask512, noduleMask512 = np.zeros((lungImg.shape[0], 512, 512)), \
                                           np.zeros((lungMask.shape[0], 512, 512)), \
                                           np.zeros((noduleMask.shape[0], 512, 512))
  originalShape = lungImg.shape 
  for z in range(lungImg.shape[0]):
    offset = (512 - originalShape[1])
    upperOffset = np.round(offset/2)
    lowerOffset = offset - upperOffset
    newOrigin = voxel2world([-upperOffset,-lowerOffset,0],origin,newSpacing)
    lungImg512[z, upperOffset:-lowerOffset,upperOffset:-lowerOffset] = lungImg[z,:,:]
    lungMask512[z, upperOffset:-lowerOffset,upperOffset:-lowerOffset] = lungMask[z,:,:]
    noduleMask512[z, upperOffset:-lowerOffset,upperOffset:-lowerOffset] = noduleMask[z,:,:]
  lungImg512 = lungImg512.astype(np.int16)
  lungMask512 = lungMask512.astype(np.int16)
  noduleMask512 = noduleMask512.astype(np.int16)
  if not os.path.exists(opts.outpath): os.makedirs(opts.outpath)
  fnImg = opts.outpath + '/' + imageName + '_lungImg.dat'
  fnLung = opts.outpath + '/' + imageName + '_lungMask.dat'
  fnNodule = opts.outpath + '/' + imageName + '_noduleMask.dat'
  fnI = np.memmap(fnImg,dtype='int16',mode='w+',shape=lungImg512.shape)
  fnL = np.memmap(fnLung,dtype='int16',mode='w+',shape=lungMask512.shape)
  fnN = np.memmap(fnNodule,dtype='int16',mode='w+',shape=noduleMask512.shape)
  fnI[:,:,:] = lungImg512[:,:,:]
  fnL[:,:,:] = lungMask512[:,:,:]
  fnN[:,:,:] = noduleMask512[:,:,:]
  del fnI
  del fnL
  del fnN       


def diceCoef(ytrue, ypred):
  smooth = 1.
  ytrue_f = K.flatten(ytrue)
  ypred_f = K.flatten(ypred)
  intersection = K.sum(ytrue_f * ypred_f)
  return (2. * intersection + smooth) / (K.sum(ytrue_f) + K.sum(ypred_f) + smooth)


def diceCoefnp(ytrue, ypred):
  smooth = 1. 
  ytruef = ytrue.flatten()
  ypredf = ypred.flatten()
  intersection = np.sum(ytruef*ypredf)
  return (2.*intersection+smooth)/(np.sum(ytruef)+np.sum(ypredf)+smooth)


def unetModel():
  K.set_image_dim_ordering('th')
  inputs = Input((1, 512, 512))
  #l1
  conv1 = Convolution2D(64, 3, 3, activation='relu', border_mode='same')(inputs)
  conv1 = Dropout(0.2)(conv1)
  conv1 = Convolution2D(64, 3, 3, activation='relu', border_mode='same')(conv1)
  pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
  #l2
  conv2 = Convolution2D(128, 3, 3, activation='relu', border_mode='same')(pool1)
  conv2 = Dropout(0.2)(conv2)
  conv2 = Convolution2D(128, 3, 3, activation='relu', border_mode='same')(conv2)
  pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
  #l3
  conv3 = Convolution2D(256, 3, 3, activation='relu', border_mode='same')(pool2)
  conv3 = Dropout(0.2)(conv3)
  conv3 = Convolution2D(256, 3, 3, activation='relu', border_mode='same')(conv3)
  pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
  #l4
  conv4 = Convolution2D(512, 3, 3, activation='relu', border_mode='same')(pool3)
  conv4 = Dropout(0.2)(conv4)
  conv4 = Convolution2D(512, 3, 3, activation='relu', border_mode='same')(conv4)
  pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)
  #l5
  conv5 = Convolution2D(1024, 3, 3, activation='relu', border_mode='same')(pool4)
  conv5 = Dropout(0.2)(conv5)
  conv5 = Convolution2D(1024, 3, 3, activation='relu', border_mode='same')(conv5)
  #l6
  up6 = merge([UpSampling2D(size=(2, 2))(conv5), conv4], mode='concat', concat_axis=1)
  conv6 = Convolution2D(512, 3, 3, activation='relu', border_mode='same')(up6)
  conv6 = Dropout(0.2)(conv6)
  conv6 = Convolution2D(512, 3, 3, activation='relu', border_mode='same')(conv6)
  #l7
  up7 = merge([UpSampling2D(size=(2, 2))(conv6), conv3], mode='concat', concat_axis=1)
  conv7 = Convolution2D(256, 3, 3, activation='relu', border_mode='same')(up7)
  conv7 = Dropout(0.2)(conv7)
  conv7 = Convolution2D(256, 3, 3, activation='relu', border_mode='same')(conv7)
  #l8
  up8 = merge([UpSampling2D(size=(2, 2))(conv7), conv2], mode='concat', concat_axis=1)
  conv8 = Convolution2D(128, 3, 3, activation='relu', border_mode='same')(up8)
  conv8 = Dropout(0.2)(conv8)
  conv8 = Convolution2D(128, 3, 3, activation='relu', border_mode='same')(conv8)
  #l9
  up9 = merge([UpSampling2D(size=(2, 2))(conv8), conv1], mode='concat', concat_axis=1)
  conv9 = Convolution2D(64, 3, 3, activation='relu', border_mode='same')(up9)
  conv9 = Dropout(0.2)(conv9)
  conv9 = Convolution2D(64, 3, 3, activation='relu', border_mode='same')(conv9)
  #l10
  conv10 = Convolution2D(1, 1, 1, activation='sigmoid')(conv9)
  #model
  model = Model(input=inputs, output=conv10)
  model.compile(optimizer=Adam(lr=1e-5), loss=diceCoef, metrics=[diceCoef]) 
  return model


def train(path, useExisting):
  print '-'*20
  print 'LOADING TRAIN DATA'
  print'-'*20
  imgsTrain = np.load(path+"trainImages.npy").astype(np.float32)
  imgsMaskTrain = np.load(path+"trainMasks.npy").astype(np.float32)
  imgsTest = np.load(path+"testImages.npy").astype(np.float32)
  imgsMaskTestTrue = np.load(path+"testMasks.npy").astype(np.float32)
  mean = np.mean(imgsTrain)  
  std = np.std(imgsTrain)  
  imgsTrain -= mean  
  imgsTrain /= std
  print '-'*20
  print 'CREATING MODEL'
  print '-'*20
  model = unetModel()
  modelCheckpoint = ModelCheckpoint('unet.hdf5', monitor='loss', save_best_only=True)
  if useExisting:
    model.load_weights('./unet.hdf5')
  print '-'*20
  print 'TRAINING MODEL' 
  print '-'*20
  model.fit(imgsTrain, imgsMaskTrain, batch_size=2, nb_epoch=20, verbose=1, shuffle=True,
            callbacks=[model_checkpoint])
  print '-'*20
  print 'LOADING SAVED WEIGHTS'
  print '-'*20
  model.load_weights('./unet.hdf5')
  print '-'*20
  print 'PREDICTING MASKS ON TEST DATA'
  print '-'*20
  numTest = len(imgsTest)
  imgsMaskTest = np.ndarray([numTest,1,512,512],dtype=np.float32)
  for i in range(numTest):
    imgsMaskTest[i] = model.predict([imgsTest[i:i+1]], verbose=0)[0]
  np.save('masksTestPredicted.npy', imgsMaskTest)
  mean = 0.0
  for i in range(numTest):
    mean+=diceCoefnp(imgsMaskTestTrue[i,0], imgsMaskTest[i,0])
  mean/=numTest
  print "MEAN DICE COEF: " + str(mean)


def getMasks(opts):
  for parent, subdir, files in os.walk(LUNA_PATH):
    flist = glob(parent+'/*.mhd')  
    if len(flist): 
      df = pd.read_csv(LUNA_AUX_PATH+'annotations.csv')
      df['file'] = df['seriesuid'].map(lambda fname: getFname(flist, fname))
      df = df.dropna()
      for fcount, f in enumerate(tqdm(flist)):
        idf = df[df['file']==f]
        if idf.shape[0]>0:
          itk = sitk.ReadImage(f)
          img = sitk.GetArrayFromImage(itk)
          z, h, w = img.shape
          origin = np.array(itk.GetOrigin())
          spacing = np.array(itk.GetSpacing())
          for node, row in idf.iterrows():
            ndx = row['coordX']
            ndy = row['coordY']
            ndz = row['coordZ']
            d = row['diameter_mm']
            imgs = np.ndarray([3,h,w],dtype=np.float32)
            masks = np.ndarray([3,h,w],dtype=np.uint8)
            center = np.array([ndx,ndy,ndz])
            vcenter = np.rint((center-origin)/spacing) 
            for i, j in enumerate(np.arange(int(vcenter[2])-1, 
                        int(vcenter[2])-2).clip(0, z-1)):
              mask = makeMask(center, d, j*spacing[2]+origin[2], 
                      w, h, spacing, origin)
              masks[i] = mask
              imgs[i] = img[j]
            #Deprecated??
            imgs[np.isnan(imgs)] = 0.
            imgs[imgs > 1e10] = 0.
            imgs[np.isinf(imgs)] = 0.
            if np.sum(imgs) < .01: print "EXTREME LOW VALUES!!"
            if imgs.any() > 1e9: print "EXTREME HIGH VALUES!!"
            np.save(os.path.join(LUNA_OUT_PATH,'images_%03d_%03d.npy' % (fcount, node)), imgs)
            np.save(os.path.join(LUNA_OUT_PATH,'masks_%03d_%03d.npy' % (fcount, node)), masks)
  print "FINISHED GENERATING LUNA MASKS"
  return True


def procMasks(fileList):
  outImages = []
  outNodemasks = []
  for fname in fileList:
    print "working on ", fname
    imgstoprocess = np.load(fname.replace("lungmask","images"))
    masks = np.load(fname)
    nodeMasks = np.load(fname.replace("lungmask","masks"))
    for i in range(len(imgstoprocess)):
      mask = masks[i]
      nodeMask = nodeMasks[i]
      img = imgstoprocess[i]
      #newsize = [512,512] 
      img= mask*img         
      newmean = np.mean(img[mask>0])  
      newstd = np.std(img[mask>0])
      oldmin = np.min(img)       
      img[img==oldmin] = newmean-1.2*newstd   
      img = img-newmean
      img = img/newstd
      labels = measure.label(mask)
      regions = measure.regionprops(labels)
      minrow = 512
      maxrow = 0
      mincol = 512
      maxcol = 0
      for prop in regions:
        b = prop.bbox
        if minrow > b[0]:
          minrow = b[0]
        if mincol > b[1]:
          mincol = b[1]
        if maxrow < b[2]:
          maxrow = b[2]
        if maxcol < b[3]:
          maxcol = b[3]
      w = maxcol-mincol
      h = maxrow-minrow
      if w > h:
        maxrow=minrow+w
      else:
        maxcol = mincol+h
      img = img[minrow:maxrow,mincol:maxcol]
      mask =  mask[minrow:maxrow,mincol:maxcol]
      if maxrow-minrow <5 or maxcol-mincol<5:  
        pass
      else:
        mean = np.mean(img)
        img = img - mean
        min = np.min(img)
        max = np.max(img)
        img = img/(max-min)
        newimage = resize(img,[512,512])
        newNodeMask = resize(nodeMask[minrow:maxrow,mincol:maxcol],[512,512])
        outImages.append(newimage)
        outNodemasks.append(newNodeMask)
  return outImages, outNodemasks


def procOutput(outImages, outNodemasks, path):
  nimages = len(outImages)
  fimages = np.ndarray([nimages,1,512,512],dtype=np.float32)
  fmasks = np.ndarray([nimages,1,512,512],dtype=np.float32)
  for i in range(nimages):
      fimages[i,0] = outImages[i]
      fmasks[i,0] = outNodemasks[i]
  rand = np.random.choice(np.asarray(range(nimages)),size=nimages,replace=False)
  test = int(0.2*nimages)
  np.save(path+"trainImages.npy",fimages[rand[test:]])
  np.save(path+"trainMasks.npy",fmasks[rand[test:]])
  np.save(path+"testImages.npy",fimages[rand[:test]])
  np.save(path+"testMasks.npy",fmasks[rand[:test]])


def procImages(path, flist):
  print flist
  print path
  for ifile in flist:
    images = np.load(ifile).astype(np.float64) 
    print "image: ", ifile
    for i in range(len(images)):
      img = images[i]
      mean = np.mean(img)
      std = np.std(img)
      img = img-mean
      if std > 0.:
        img = img/std
      middle = img[100:400,100:400] 
      mean = np.mean(middle)  
      max = np.max(img)
      min = np.min(img)
      img[img==max]=mean
      img[img==min]=mean
      kmeans = KMeans(n_clusters=2).fit(np.reshape(middle,[np.prod(middle.shape),1]))
      centers = sorted(kmeans.cluster_centers_.flatten())
      threshold = np.mean(centers)
      threshimg = np.where(img<threshold,1.0,0.0)  
      eroded = morphology.erosion(threshimg,np.ones([4,4]))
      dilation = morphology.dilation(eroded,np.ones([10,10]))
      labels = measure.label(dilation)
      #labelVals = np.unique(labels)
      regions = measure.regionprops(labels) #was labels, not label_vals
      goodLabels = []
      for prop in regions:
        b = prop.bbox
        if b[2]-b[0]<475 and b[3]-b[1]<475 and b[0]>40 and b[2]<472:
          goodLabels.append(prop.label)
      mask = np.ndarray([512,512],dtype=np.int8)
      mask[:] = 0
      for N in goodLabels:
        mask = mask + np.where(labels==N,1,0)
      mask = morphology.dilation(mask,np.ones([10,10])) 
      images[i] = mask
    np.save(ifile.replace("images","lungmask"),images)
  return True


def trainSegmentation(opts):
  print "\n\nGETTING MASKS\n\n"
  if getMasks(opts):
    print "\n\nPROCESSING LUNA IMAGES\n\n"
    if procImages(LUNA_OUT_PATH, glob(LUNA_OUT_PATH+'/images_*.npy')):
      print "\n\nPROCESSING LUNA MASKS\n\n"
      images, nodemasks = procMasks(glob(LUNA_OUT_PATH+'/masks_*.npy'))
      procOutput(images, nodemasks, LUNA_OUT_PATH)
      print "\n\nTRAINING UNET SEGMENTATION\n\n"
      train(LUNA_OUT_PATH, useExisting=False)
      print "\n\nSEGMENTATION TRAINING COMPLETED\n\n"


if __name__ == '__main__':
  trainSegmentation(opts=None)


