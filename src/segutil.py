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

LUNA_PATH = '/hdfs/a/subset9/'
LUNA_AUX_PATH = '/hdfs/a/CSVFILES/'
LUNA_OUT_PATH = '/hdfs/a/nodules/segimage/nine/images/'
#XMLDIR = '/home/rrg0013@auburn.edu/xmlc/'
DEBUG = False


scanids = []#'bin0'
''','1.3.6.1.4.1.14519.5.2.1.6279.6001.534083630500464995109143618896'\
           ,'1.3.6.1.4.1.14519.5.2.1.6279.6001.868211851413924881662621747734'\
           ,'1.3.6.1.4.1.14519.5.2.1.6279.6001.450501966058662668272378865145'\
           ,'1.3.6.1.4.1.14519.5.2.1.6279.6001.295420274214095686326263147663'\
           ,'1.3.6.1.4.1.14519.5.2.1.6279.6001.187451715205085403623595258748'\
           ,''\
           ,'bin1'\
           ,'1.3.6.1.4.1.14519.5.2.1.6279.6001.163994693532965040247348251579'\
           ,'1.3.6.1.4.1.14519.5.2.1.6279.6001.169128136262002764211589185953'\
           ,'1.3.6.1.4.1.14519.5.2.1.6279.6001.226456162308124493341905600418'\
           ,"1.3.6.1.4.1.14519.5.2.1.6279.6001.315214756157389122376518747372"\
           ,'1.3.6.1.4.1.14519.5.2.1.6279.6001.247769845138587733933485039556'\
           ,'1.3.6.1.4.1.14519.5.2.1.6279.6001.275766318636944297772360944907'\
           ,'1.3.6.1.4.1.14519.5.2.1.6279.6001.287966244644280690737019247886'\
           ,'1.3.6.1.4.1.14519.5.2.1.6279.6001.325164338773720548739146851679'\
           ,'1.3.6.1.4.1.14519.5.2.1.6279.6001.655242448149322898770987310561'\
           ,'1.3.6.1.4.1.14519.5.2.1.6279.6001.674809958213117379592437424616'\
           ,'1.3.6.1.4.1.14519.5.2.1.6279.6001.756684168227383088294595834066'\
           ,'1.3.6.1.4.1.14519.5.2.1.6279.6001.861997885565255340442123234170'\
           ,'1.3.6.1.4.1.14519.5.2.1.6279.6001.935683764293840351008008793409'\
           ,"1.3.6.1.4.1.14519.5.2.1.6279.6001.952265563663939823135367733681"\
           ,''\
           ,'bin2'\
           ,'1.3.6.1.4.1.14519.5.2.1.6279.6001.133378195429627807109985347209'\
           ,'1.3.6.1.4.1.14519.5.2.1.6279.6001.144883090372691745980459537053'\
           ,'1.3.6.1.4.1.14519.5.2.1.6279.6001.199975006921901879512837687266'\
           ,'1.3.6.1.4.1.14519.5.2.1.6279.6001.192256506776434538421891524301'\
           ,'1.3.6.1.4.1.14519.5.2.1.6279.6001.217955041973656886482758642958'\
           ,'1.3.6.1.4.1.14519.5.2.1.6279.6001.235364978775280910367690540811'\
           ,'1.3.6.1.4.1.14519.5.2.1.6279.6001.943403138251347598519939390311'\
           ,''\
           ,'bin3'\
           ,'1.3.6.1.4.1.14519.5.2.1.6279.6001.171667800241622018839592854574'\
           ,'1.3.6.1.4.1.14519.5.2.1.6279.6001.244204120220889433826451158706'\
           ,'1.3.6.1.4.1.14519.5.2.1.6279.6001.272190966764020277652079081128'\
           ,'1.3.6.1.4.1.14519.5.2.1.6279.6001.277662902666135640561346462196'\
           ,'1.3.6.1.4.1.14519.5.2.1.6279.6001.481278873893653517789960724156'\
           ,'1.3.6.1.4.1.14519.5.2.1.6279.6001.842317928015463083368074520378'\
           ,'1.3.6.1.4.1.14519.5.2.1.6279.6001.319066480138812986026181758474'\
           ,'1.3.6.1.4.1.14519.5.2.1.6279.6001.975426625618184773401026809852'\
           ,''\
           ,'bin4'\
           ,'1.3.6.1.4.1.14519.5.2.1.6279.6001.122914038048856168343065566972'\
           ,'1.3.6.1.4.1.14519.5.2.1.6279.6001.142154819868944114554521645782'\
           ,'1.3.6.1.4.1.14519.5.2.1.6279.6001.161855583909753609742728521805'\
           ,'1.3.6.1.4.1.14519.5.2.1.6279.6001.205993750485568250373835565680'\
           ,'1.3.6.1.4.1.14519.5.2.1.6279.6001.312704771348460502013249647868'\
           ,'1.3.6.1.4.1.14519.5.2.1.6279.6001.242761658169703141430370511586'\
           ,'1.3.6.1.4.1.14519.5.2.1.6279.6001.897684031374557757145405000951'\
           ,'1.3.6.1.4.1.14519.5.2.1.6279.6001.211956804948320236390242845468'\
           ,'1.3.6.1.4.1.14519.5.2.1.6279.6001.211051626197585058967163339846'\
           ,'1.3.6.1.4.1.14519.5.2.1.6279.6001.228511122591230092662900221600'\
           ,'1.3.6.1.4.1.14519.5.2.1.6279.6001.230416590143922549745658357505'\
           ,'1.3.6.1.4.1.14519.5.2.1.6279.6001.337845202462615014431060697507'\
           ,''\
           ,'bin5'\
           ,''\
           ,'1.3.6.1.4.1.14519.5.2.1.6279.6001.111258527162678142285870245028'\
           ,'1.3.6.1.4.1.14519.5.2.1.6279.6001.112740418331256326754121315800'\
           ,'1.3.6.1.4.1.14519.5.2.1.6279.6001.138904664700896606480369521124'\
           ,'1.3.6.1.4.1.14519.5.2.1.6279.6001.176638348958425792989125209419'\
           ,'1.3.6.1.4.1.14519.5.2.1.6279.6001.296738183013079390785739615169'\
           ,'1.3.6.1.4.1.14519.5.2.1.6279.6001.338104567770715523699587505022'\
           ,'1.3.6.1.4.1.14519.5.2.1.6279.6001.338875090785618956575597613546'\
           ,'1.3.6.1.4.1.14519.5.2.1.6279.6001.725023183844147505748475581290'\
           ,''\
           ,'bin6'\
           ,'1.3.6.1.4.1.14519.5.2.1.6279.6001.173556680294801532247454313511'\
           ,'1.3.6.1.4.1.14519.5.2.1.6279.6001.247816269490470394602288565775'\
           ,'1.3.6.1.4.1.14519.5.2.1.6279.6001.286217539434358186648717203667'\
           ,'1.3.6.1.4.1.14519.5.2.1.6279.6001.167237290696350215427953159586'\
           ,'1.3.6.1.4.1.14519.5.2.1.6279.6001.315187221221054114974341475212'\
           ,'1.3.6.1.4.1.14519.5.2.1.6279.6001.316900421002460665752357657094'\
           ,'1.3.6.1.4.1.14519.5.2.1.6279.6001.329404588567903628160652715124'\
           ,'1.3.6.1.4.1.14519.5.2.1.6279.6001.330544495001617450666819906758'\
           ,'1.3.6.1.4.1.14519.5.2.1.6279.6001.338447145504282422142824032832'\
           ,'1.3.6.1.4.1.14519.5.2.1.6279.6001.948414623428298219623354433437'\
           ,'1.3.6.1.4.1.14519.5.2.1.6279.6001.955688628308192728558382581802'\
           ,'1.3.6.1.4.1.14519.5.2.1.6279.6001.618434772073433276874225174904'\
           ,'1.3.6.1.4.1.14519.5.2.1.6279.6001.321935195060268166151738328001'\
           ,'1.3.6.1.4.1.14519.5.2.1.6279.6001.658611160253017715059194304729'\
           ,''\
           ,'bin7'\
           ,'1.3.6.1.4.1.14519.5.2.1.6279.6001.113679818447732724990336702075'\
           ,'1.3.6.1.4.1.14519.5.2.1.6279.6001.122621219961396951727742490470'\
           ,'1.3.6.1.4.1.14519.5.2.1.6279.6001.125124219978170516876304987559'\
           ,'1.3.6.1.4.1.14519.5.2.1.6279.6001.214252223927572015414741039150'\
           ,'1.3.6.1.4.1.14519.5.2.1.6279.6001.242624386080831911167122628616'\
           ,'1.3.6.1.4.1.14519.5.2.1.6279.6001.215640837032688688030770057224'\
           ,'1.3.6.1.4.1.14519.5.2.1.6279.6001.294120933998772507043263238704'\
           ,'1.3.6.1.4.1.14519.5.2.1.6279.6001.613212850444255764524630781782'\
           ,'1.3.6.1.4.1.14519.5.2.1.6279.6001.323535944958374186208096541480'\
           ,''\
           ,'bin8'\
           ,'1.3.6.1.4.1.14519.5.2.1.6279.6001.149041668385192796520281592139'\
           ,'1.3.6.1.4.1.14519.5.2.1.6279.6001.172243743899615313644757844726'\
           ,'1.3.6.1.4.1.14519.5.2.1.6279.6001.204287915902811325371247860532'\
           ,'1.3.6.1.4.1.14519.5.2.1.6279.6001.202283133206014258077705539227'\
           ,'1.3.6.1.4.1.14519.5.2.1.6279.6001.225515255547637437801620523312'\
           ,'1.3.6.1.4.1.14519.5.2.1.6279.6001.249314567767437206995861966896'\
           ,'1.3.6.1.4.1.14519.5.2.1.6279.6001.257383535269991165447822992959'\
           ,'1.3.6.1.4.1.14519.5.2.1.6279.6001.297988578825170426663869669862'\
           ,'1.3.6.1.4.1.14519.5.2.1.6279.6001.478062284228419671253422844986'\
           ,'1.3.6.1.4.1.14519.5.2.1.6279.6001.776800177074349870648765614630'\
           ,''\
           ,'bin9'\
           ,'1.3.6.1.4.1.14519.5.2.1.6279.6001.173931884906244951746140865701'\
           ,'1.3.6.1.4.1.14519.5.2.1.6279.6001.188619674701053082195613114069'\
           ,'1.3.6.1.4.1.14519.5.2.1.6279.6001.195557219224169985110295082004'\
           ,'1.3.6.1.4.1.14519.5.2.1.6279.6001.300270516469599170290456821227'\
           ,'1.3.6.1.4.1.14519.5.2.1.6279.6001.340158437895922179455019686521'\
           ,'1.3.6.1.4.1.14519.5.2.1.6279.6001.387954549120924524005910602207'\
           ,'1.3.6.1.4.1.14519.5.2.1.6279.6001.392861216720727557882279374324']'''


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
  for nodule in scan.clusters:
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


def getROIs(image, mask, origin, annotations):
  #print "GETTING ROIS"
  irois, mrois = [], []
  #print annotations.shape
  for nodule in annotations:
    #print nodule.shape
    coords = np.round(np.array(abs(nodule[:3]-origin))).astype(int)
    z = coords[2]
    y = coords[1]
    x = coords[0]
    if image[z-32:z+32,y-32:y+32,x-32:x+32].shape != (64,64,64):
      continue
    #  if z-32 < 0:

    #  elif shape[0]-z < 32:

    #  if y-32 < 0:

    #  elif shape[1]-y < 32:

    #  if x-32 < 0:

    #  elif shape[2]-x < 32:
    
    #else:
    iroi = mask[z-32:z+32,y-32:y+32,x-32:x+32]  
    mroi = mask[z-32:z+32,y-32:y+32,x-32:x+32]
    if np.sum(mroi[:,:,0]) > 0:
      print '\n\n\nERROR\n\nMASK ERROR\n\nERROR\n\n\n'
      continue
    mroi = flattenImage(mroi)
    mrois.append(mroi)
    #iroi = image[z-32:z+32,y-32:y+32,x-32:x+32]
    iroi = flattenImage(iroi)
    irois.append(iroi)
  return irois, mrois


'''def getMasks():
  count = 0
  flag = False
  flagb = False
  for parent, subdir, files in os.walk(LUNA_PATH):
    df = pd.read_csv(LUNA_AUX_PATH+'annotations.csv')
    for fname in files:
      if ".mhd" in fname.lower():
        print fname
        if scanids is not None:
          for scan in scanids:
            flag = True
            fno = fname[:-4]
            if fno == scan:
              flag = False
              break
        #if flag:
        #  flag = False
        #  continue
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
        origin = np.array(itk.GetOrigin())
        #ispacing = np.array(itk.GetSpacing())
        #icoords = segment.getCoords(annotations,iorigin,ispacing)
        print fname
        print coords
        #scan = lidc.xml(XMLDIR,fname,ispacing,iorigin)
        #scan.annotations = coords
        #lidc.cluster(scan)
        #scan.combine()
        #scan.validateNodules()
        #mask = getImageMaskROI(scan,image.shape)
        #mask = getImageMask(image.shape,origin,annotations)
        annotations = annotations[['coordX','coordY','coordZ','diameter_mm']].as_matrix()
        mask = getImageMask(image.shape,origin,annotations)       
        if annotations.shape[0] == 0: continue
        else:
          if flag:
            continue
        image = scipy.ndimage.interpolation.zoom(image, szfactor, mode='nearest')
        mask = scipy.ndimage.interpolation.zoom(mask, szfactor, mode='nearest')
        mask[mask>5e-1]=1
        mask[mask<=5e-1]=0
        irois, mrois = getROIs(image, mask, origin, annotations)
        if len(irois) > 0 and len(mrois) > 0:
          #print "SAVING"
          for i in range(len(irois)):
            mroi = mrois[i]
            iroi = irois[i]
            fname = LUNA_OUT_PATH+'/'+str(count)+'_image.png'
            scipy.misc.toimage(iroi[:,:],cmin=-1000,cmax=1000).save(fname)
            fname = LUNA_OUT_PATH+'/'+str(count)+'_mask.png'
            scipy.misc.toimage(mroi[:,:],cmin=0,cmax=1).save(fname)
            count += 1
        print "TOTAL: " + str(count)'''


def getMasksRecover():
  count = 0
  for parent, subdir, files in os.walk(LUNA_PATH):
    df = pd.read_csv(LUNA_AUX_PATH+'annotations.csv')
    for fname in files:
      if ".mhd" in fname.lower():
        '''print fname
        if scanids is not None:
          for scan in scanids:
            flag = True
            fno = fname[:-4]
            if fno == scan:
              flag = False
              break'''
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
        annotations = annotations[['coordX','coordY','coordZ','diameter_mm']].as_matrix()
        if annotations.shape[0] == 0: continue
        #else: 
        #  if flag: continue
        image = scipy.ndimage.interpolation.zoom(image, szfactor, mode='nearest')
        origin = np.array(itk.GetOrigin())
        mask = getImageMask(image.shape, origin, annotations)
        irois, mrois = getROIs(image, mask, origin, annotations)
        if len(irois) > 0 and len(mrois) > 0:
          print "PRINTING IMAGES & MASKS"
          for i in range(len(irois)):
            mroi = mrois[i]
            iroi = irois[i]
            fname = LUNA_OUT_PATH+'/'+str(count)+'_image.png'
            scipy.misc.toimage(iroi[:,:],cmin=-1000,cmax=1000).save(fname)
            fname = LUNA_OUT_PATH+'/'+str(count)+'_mask.png'
            scipy.misc.toimage(mroi[:,:],cmin=0,cmax=1).save(fname)
            count += 1
        print "TOTAL: " + str(count)



if __name__ == '__main__':
  #getMasks()
  getMasksRecover()

