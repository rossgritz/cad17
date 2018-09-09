import sys

sys.path.append('/home/admin/git/nodules/src/')
sys.path.append('/home/admin/git/nodules/keras/')
sys.path.append('/home/admin/git/nodules/models/')

import os
import copy
import time
import math
import skimage
import scipy.misc
import numpy as np
import pandas as pd
import SimpleITK as sitk
import matplotlib.pyplot as plt
import scipy.ndimage

import seg3d 
#import seg3dB1 as seg3d
import candidates as cd
import candgen as cg

from skimage import feature
from skimage import measure
from glob import glob

import tensorflow as tf 
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction=0.45
set_session(tf.Session(config=config))

#TEST = 4
#VAL = 0
#MODEL = 'lowvallossbest'#36  
BLUR1 = False

#DATAPATH =  '/hdfs/a/subset'+str(VAL)+'/'
ANNOTATIONS = '/home/admin/git/nodules/resources/annotations.csv'
#WEIGHTSPATH = '/hdfs/b/nodules/segmodels/'+str(TEST)+'/'+str(VAL)+'/'+str(MODEL)+'.h5'
#OUTPATH = '/hdfs/b/nodules/segmodels/'+str(TEST)+'/'+str(VAL)+'/'+str(MODEL)+'/'
#OUTFILE = '/hdfs/b/nodules/segmodels/'+str(TEST)+'/'+str(VAL)+'/'+str(MODEL)+'.txt'

MAXCANDS = 10000
THRESH = 9.9e-2
COUNT = 0
FCOUNT = 0

START=0
STRIDE = 32
if STRIDE == 16: SIXTEEN = True
else: SIXTEEN = False
scanids = None
scanids = ['bin0'\
           ,'1.3.6.1.4.1.14519.5.2.1.6279.6001.430109407146633213496148200410'\
           ,'1.3.6.1.4.1.14519.5.2.1.6279.6001.534083630500464995109143618896'\
           ,'1.3.6.1.4.1.14519.5.2.1.6279.6001.868211851413924881662621747734'\
           ,'1.3.6.1.4.1.14519.5.2.1.6279.6001.450501966058662668272378865145'\
           ,'1.3.6.1.4.1.14519.5.2.1.6279.6001.295420274214095686326263147663'\
           ,'1.3.6.1.4.1.14519.5.2.1.6279.6001.187451715205085403623595258748'\
           ,'1.3.6.1.4.1.14519.5.2.1.6279.6001.213140617640021803112060161074'\
           ,'1.3.6.1.4.1.14519.5.2.1.6279.6001.130438550890816550994739120843'\
           ,'1.3.6.1.4.1.14519.5.2.1.6279.6001.417815314896088956784723476543'\
           ,'1.3.6.1.4.1.14519.5.2.1.6279.6001.138080888843357047811238713686'\
           ,'1.3.6.1.4.1.14519.5.2.1.6279.6001.227962600322799211676960828223'\
#           ,'1.3.6.1.4.1.14519.5.2.1.6279.6001.898642529028521482602829374444'\
#           ,'1.3.6.1.4.1.14519.5.2.1.6279.6001.216882370221919561230873289517'\
#           ,'1.3.6.1.4.1.14519.5.2.1.6279.6001.534006575256943390479252771547'\
           ,'bin1'\
           ,'1.3.6.1.4.1.14519.5.2.1.6279.6001.145759169833745025756371695397'\
           ,'1.3.6.1.4.1.14519.5.2.1.6279.6001.183184435049555024219115904825'\
           ,'1.3.6.1.4.1.14519.5.2.1.6279.6001.163994693532965040247348251579'\
           ,'1.3.6.1.4.1.14519.5.2.1.6279.6001.169128136262002764211589185953'\
           ,'1.3.6.1.4.1.14519.5.2.1.6279.6001.226456162308124493341905600418'\
           ,"1.3.6.1.4.1.14519.5.2.1.6279.6001.315214756157389122376518747372"\
           ,'1.3.6.1.4.1.14519.5.2.1.6279.6001.247769845138587733933485039556'\
           ,'1.3.6.1.4.1.14519.5.2.1.6279.6001.275766318636944297772360944907'\
           ,'1.3.6.1.4.1.14519.5.2.1.6279.6001.287966244644280690737019247886'\
           ,'1.3.6.1.4.1.14519.5.2.1.6279.6001.325164338773720548739146851679'\
           ,'1.3.6.1.4.1.14519.5.2.1.6279.6001.479402560265137632920333093071'\
           ,'1.3.6.1.4.1.14519.5.2.1.6279.6001.655242448149322898770987310561'\
           ,'1.3.6.1.4.1.14519.5.2.1.6279.6001.674809958213117379592437424616'\
           ,'1.3.6.1.4.1.14519.5.2.1.6279.6001.756684168227383088294595834066'\
           ,'1.3.6.1.4.1.14519.5.2.1.6279.6001.861997885565255340442123234170'\
           ,'1.3.6.1.4.1.14519.5.2.1.6279.6001.935683764293840351008008793409'\
           ,"1.3.6.1.4.1.14519.5.2.1.6279.6001.952265563663939823135367733681"\
           ,'bin2'\
           ,'1.3.6.1.4.1.14519.5.2.1.6279.6001.133378195429627807109985347209'\
           ,'1.3.6.1.4.1.14519.5.2.1.6279.6001.144883090372691745980459537053'\
           ,'1.3.6.1.4.1.14519.5.2.1.6279.6001.199975006921901879512837687266'\
           ,'1.3.6.1.4.1.14519.5.2.1.6279.6001.192256506776434538421891524301'\
           ,'1.3.6.1.4.1.14519.5.2.1.6279.6001.217955041973656886482758642958'\
           ,'1.3.6.1.4.1.14519.5.2.1.6279.6001.235364978775280910367690540811'\
           ,'1.3.6.1.4.1.14519.5.2.1.6279.6001.943403138251347598519939390311'\
           ,'bin3'\
           ,'1.3.6.1.4.1.14519.5.2.1.6279.6001.171667800241622018839592854574'\
           ,'1.3.6.1.4.1.14519.5.2.1.6279.6001.244204120220889433826451158706'\
           ,'1.3.6.1.4.1.14519.5.2.1.6279.6001.272190966764020277652079081128'\
           ,'1.3.6.1.4.1.14519.5.2.1.6279.6001.277662902666135640561346462196'\
           ,'1.3.6.1.4.1.14519.5.2.1.6279.6001.481278873893653517789960724156'\
           ,'1.3.6.1.4.1.14519.5.2.1.6279.6001.842317928015463083368074520378'\
           ,'1.3.6.1.4.1.14519.5.2.1.6279.6001.319066480138812986026181758474'\
           ,'1.3.6.1.4.1.14519.5.2.1.6279.6001.975426625618184773401026809852'\
           ,'1.3.6.1.4.1.14519.5.2.1.6279.6001.254254303842550572473665729969'\
           ,'1.3.6.1.4.1.14519.5.2.1.6279.6001.275007193025729362844652516689'\
           ,'bin4'\
           ,'1.3.6.1.4.1.14519.5.2.1.6279.6001.122914038048856168343065566972'\
           ,'1.3.6.1.4.1.14519.5.2.1.6279.6001.142154819868944114554521645782'\
           ,'1.3.6.1.4.1.14519.5.2.1.6279.6001.161855583909753609742728521805'\
           ,'1.3.6.1.4.1.14519.5.2.1.6279.6001.205993750485568250373835565680'\
           ,'1.3.6.1.4.1.14519.5.2.1.6279.6001.242761658169703141430370511586'\
           ,'1.3.6.1.4.1.14519.5.2.1.6279.6001.897684031374557757145405000951'\
           ,'1.3.6.1.4.1.14519.5.2.1.6279.6001.211956804948320236390242845468'\
           ,'1.3.6.1.4.1.14519.5.2.1.6279.6001.211051626197585058967163339846'\
           ,'1.3.6.1.4.1.14519.5.2.1.6279.6001.228511122591230092662900221600'\
           ,'1.3.6.1.4.1.14519.5.2.1.6279.6001.230416590143922549745658357505'\
           ,'1.3.6.1.4.1.14519.5.2.1.6279.6001.337845202462615014431060697507'\
           ,'1.3.6.1.4.1.14519.5.2.1.6279.6001.312704771348460502013249647868'\
           ,'1.3.6.1.4.1.14519.5.2.1.6279.6001.254254303842550572473665729969'\
           ,'bin5'\
           ,'1.3.6.1.4.1.14519.5.2.1.6279.6001.111258527162678142285870245028'\
           ,'1.3.6.1.4.1.14519.5.2.1.6279.6001.112740418331256326754121315800'\
           ,'1.3.6.1.4.1.14519.5.2.1.6279.6001.138904664700896606480369521124'\
           ,'1.3.6.1.4.1.14519.5.2.1.6279.6001.176638348958425792989125209419'\
           ,'1.3.6.1.4.1.14519.5.2.1.6279.6001.296738183013079390785739615169'\
           ,'1.3.6.1.4.1.14519.5.2.1.6279.6001.338104567770715523699587505022'\
           ,'1.3.6.1.4.1.14519.5.2.1.6279.6001.338875090785618956575597613546'\
           ,'1.3.6.1.4.1.14519.5.2.1.6279.6001.725023183844147505748475581290'\
           ,'1.3.6.1.4.1.14519.5.2.1.6279.6001.262736997975960398949912434623'\
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
           ,'1.3.6.1.4.1.14519.5.2.1.6279.6001.129982010889624423230394257528'\
           ,'bin8'\
           ,'1.3.6.1.4.1.14519.5.2.1.6279.6001.149041668385192796520281592139'\
           ,'1.3.6.1.4.1.14519.5.2.1.6279.6001.172243743899615313644757844726'\
           ,'1.3.6.1.4.1.14519.5.2.1.6279.6001.204287915902811325371247860532'\
           ,'1.3.6.1.4.1.14519.5.2.1.6279.6001.202283133206014258077705539227'\
           ,'1.3.6.1.4.1.14519.5.2.1.6279.6001.225515255547637437801620523312'\
           ,'1.3.6.1.4.1.14519.5.2.1.6279.6001.249314567767437206995861966896'\
           ,'1.3.6.1.4.1.14519.5.2.1.6279.6001.257383535269991165447822992959'\
           ,'1.3.6.1.4.1.14519.5.2.1.6279.6001.776800177074349870648765614630'\
           ,'1.3.6.1.4.1.14519.5.2.1.6279.6001.297988578825170426663869669862'\
           ,'1.3.6.1.4.1.14519.5.2.1.6279.6001.478062284228419671253422844986'\
           ,'bin9'\
           ,'1.3.6.1.4.1.14519.5.2.1.6279.6001.173931884906244951746140865701'\
           ,'1.3.6.1.4.1.14519.5.2.1.6279.6001.188619674701053082195613114069'\
           ,'1.3.6.1.4.1.14519.5.2.1.6279.6001.195557219224169985110295082004'\
           ,'1.3.6.1.4.1.14519.5.2.1.6279.6001.300270516469599170290456821227'\
           ,'1.3.6.1.4.1.14519.5.2.1.6279.6001.340158437895922179455019686521'\
           ,'1.3.6.1.4.1.14519.5.2.1.6279.6001.387954549120924524005910602207'\
           ,'1.3.6.1.4.1.14519.5.2.1.6279.6001.392861216720727557882279374324'\
           ,'9check-xx'\
           ,'done']
scanids =  ['0check-13'\
           ##,'1.3.6.1.4.1.14519.5.2.1.6279.6001.134996872583497382954024478441'\
           #,'1.3.6.1.4.1.14519.5.2.1.6279.6001.305858704835252413616501469037'\
           #,'1.3.6.1.4.1.14519.5.2.1.6279.6001.293757615532132808762625441831'\
           ,'1.3.6.1.4.1.14519.5.2.1.6279.6001.294188507421106424248264912111'\
           ,'1.3.6.1.4.1.14519.5.2.1.6279.6001.109002525524522225658609808059'\
           ,'1check-11'\
           #,'1.3.6.1.4.1.14519.5.2.1.6279.6001.134370886216012873213579659366'\
           ##,'1.3.6.1.4.1.14519.5.2.1.6279.6001.128881800399702510818644205032'\
           #,'1.3.6.1.4.1.14519.5.2.1.6279.6001.128059192202504367870633619224'\
           ,'1.3.6.1.4.1.14519.5.2.1.6279.6001.121824995088859376862458155637'\
           ,'1.3.6.1.4.1.14519.5.2.1.6279.6001.114218724025049818743426522343'\
           ,'2check-17'\
           #,'1.3.6.1.4.1.14519.5.2.1.6279.6001.124663713663969377020085460568'\
           #,'1.3.6.1.4.1.14519.5.2.1.6279.6001.121993590721161347818774929286'\
           ##,'1.3.6.1.4.1.14519.5.2.1.6279.6001.100621383016233746780170740405'\
           ,'1.3.6.1.4.1.14519.5.2.1.6279.6001.267519732763035023633235877753'\
           ,'1.3.6.1.4.1.14519.5.2.1.6279.6001.265453131727473342790950829556'\
           ,'3check-12'\
           ##,'1.3.6.1.4.1.14519.5.2.1.6279.6001.276556509002726404418399209377'\
           #,'1.3.6.1.4.1.14519.5.2.1.6279.6001.292057261351416339496913597985'\
           #,'1.3.6.1.4.1.14519.5.2.1.6279.6001.274052674198758621258447180130'\
           ,'1.3.6.1.4.1.14519.5.2.1.6279.6001.269075535958871753309238331179'\
           ,'1.3.6.1.4.1.14519.5.2.1.6279.6001.154703816225841204080664115280'\
           ,'4check-20'\
           ##,'1.3.6.1.4.1.14519.5.2.1.6279.6001.141511313712034597336182402384'\
           #,'1.3.6.1.4.1.14519.5.2.1.6279.6001.141430002307216644912805017227'\
           #,'1.3.6.1.4.1.14519.5.2.1.6279.6001.107351566259572521472765997306'\
           ,'1.3.6.1.4.1.14519.5.2.1.6279.6001.259123825760999546551970425757'\
           ,'1.3.6.1.4.1.14519.5.2.1.6279.6001.265775376735520890308424143898'\
           ,'5check-14'\
           ##,'1.3.6.1.4.1.14519.5.2.1.6279.6001.118140393257625250121502185026'\
           #,'1.3.6.1.4.1.14519.5.2.1.6279.6001.257840703452266097926250569223'\
           #,'1.3.6.1.4.1.14519.5.2.1.6279.6001.258220324170977900491673635112'\
           ,'1.3.6.1.4.1.14519.5.2.1.6279.6001.174907798609768549012640380786'\
           ,'1.3.6.1.4.1.14519.5.2.1.6279.6001.275755514659958628040305922764'\
           ,'6check-10'\
           ##,'1.3.6.1.4.1.14519.5.2.1.6279.6001.147325126373007278009743173696'\
           #,'1.3.6.1.4.1.14519.5.2.1.6279.6001.130765375502800983459674173881'\
           #,'1.3.6.1.4.1.14519.5.2.1.6279.6001.129007566048223160327836686225'\
           ,'1.3.6.1.4.1.14519.5.2.1.6279.6001.123654356399290048011621921476'\
           ,'1.3.6.1.4.1.14519.5.2.1.6279.6001.119209873306155771318545953948'\
           ,'7check-16'\
           ##,'1.3.6.1.4.1.14519.5.2.1.6279.6001.140253591510022414496468423138'\
           #,'1.3.6.1.4.1.14519.5.2.1.6279.6001.114249388265341701207347458535'\
           #,'1.3.6.1.4.1.14519.5.2.1.6279.6001.111496024928645603833332252962'\
           ,'1.3.6.1.4.1.14519.5.2.1.6279.6001.106379658920626694402549886949'\
           ,'1.3.6.1.4.1.14519.5.2.1.6279.6001.249404938669582150398726875826'\
           ,'8check-16'\
           ##,'1.3.6.1.4.1.14519.5.2.1.6279.6001.168833925301530155818375859047'\
           #,'1.3.6.1.4.1.14519.5.2.1.6279.6001.153732973534937692357111055819'\
           #,'1.3.6.1.4.1.14519.5.2.1.6279.6001.137773550852881583165286615668'\
           ,'1.3.6.1.4.1.14519.5.2.1.6279.6001.131939324905446238286154504249'\
           ,'1.3.6.1.4.1.14519.5.2.1.6279.6001.100225287222365663678666836860'\
           ,'9check-xx'\
           ##,'1.3.6.1.4.1.14519.5.2.1.6279.6001.134519406153127654901640638633'\
           #,'1.3.6.1.4.1.14519.5.2.1.6279.6001.124822907934319930841506266464'\
           #,'1.3.6.1.4.1.14519.5.2.1.6279.6001.114914167428485563471327801935'\
           ,'1.3.6.1.4.1.14519.5.2.1.6279.6001.112767175295249119452142211437'\
           ,'1.3.6.1.4.1.14519.5.2.1.6279.6001.230491296081537726468075344411'\
           ,'end']


def getFilelist(datapath):
  for parent, subdir, files in os.walk(datapath):
    filelist = glob(parent+'/*.mhd')
    filepathlist = copy.deepcopy(filelist)
  templist = []
  for f in filelist:
    f = f.split('/')
    f = f[-1]
    templist.append(f[:-4])
  filelist = templist
  return filelist, filepathlist


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


def getSegmentationShape(oldshape):
  factors = (np.asarray(oldshape)/32+2)*32
  return factors


def getPadding(segShape, image):
  lpad = (segShape-np.asarray(image.shape))/2
  hpad = copy.deepcopy(lpad)
  if image.shape[0]%2>0:
    hpad[0] += 1
  if image.shape[1]%2>0:
    hpad[1] += 1
  if image.shape[2]%2>0:
    hpad[2] += 1
  return lpad, hpad


##Try strides of 16 and summing ... 
def segmentImage(image, model, strides, stride=32):
  if stride==32:
    du,dv,dw=strides[0]-1,strides[1]-1,strides[2]-1
  else:
    du,dv,dw=(strides[0]-1)*2,(strides[1]-1)*2,(strides[2]-1)*2
  outshape = (du*stride,dv*stride,dw*stride)
  segmentation = np.zeros(outshape)
  print "OUTSHAPE"
  print outshape
  if stride==32:w=2
  else:
    w=4
    du-=2
    dv-=2
    dw-=2
  for i in range(du):#-it):
    for j in range(dv):#-it):
      for k in range(dw):#-it):
        roi = image[i*stride:(i+w)*stride,
                    j*stride:(j+w)*stride,
                    k*stride:(k+w)*stride]
        if BLUR1:
          roi = scipy.ndimage.filters.gaussian_filter(roi, 2., truncate=4.)
        roi = np.expand_dims(np.expand_dims(roi,axis=0),axis=0)
        output = model.predict(roi, batch_size=1)
        segmentation[i*stride:(i+w/2)*stride,
                     j*stride:(j+w/2)*stride,
                     k*stride:(k+w/2)*stride] += output[0,0]
  return segmentation


def writeCand(roi,uid,outpath,props,nodule=False):
  bb = np.zeros((512,512))
  if roi.shape != (64,64,64): 
    print "ORIGINAL SHAPE " + str(roi.shape)
    print "ERROR ROI"
    return
  for i in range(8):
    for j in range(8):
      bb[i*64:(i+1)*64,j*64:(j+1)*64] = roi[i*8+j,:,:]
  if nodule:
    fname = 't_'+props+str(uid)+'.png'
  else:
    fname = 'f_'+props+str(uid)+'.png'
  if not os.path.exists(outpath): os.makedirs(outpath)
  fname = outpath + '/' + fname
  scipy.misc.toimage(bb[:,:],cmin=-1000,cmax=1000).save(fname)


def getDistance(a,b):
  du = (a[0]-b[0])
  dv = (a[1]-b[1])
  dw = (a[2]-b[2])
  return math.sqrt(du*du+dv*dv+dw*dw)

 
def assess(outfile, labels, coords, seriesuid, outpath, pseg, image, diameter):
  if not os.path.exists(outpath): os.makedirs(outpath)
  f = open(outfile,'a+')
  correct = []
  label = np.zeros(labels.shape).astype(int)
  global FCOUNT
  global COUNT

  for i, coord in enumerate(coords):
    c3,c2,c1 = coord
    ff = True
    l = 0
    for a in range(3):
      for b in range(3):
        for c in range(3):
          _a = -1 + a
          _b = -1 + b
          _c = -1 + c
          l = labels[c1+_a,c2+_b,c3+_c]
          if l != 0:
            COUNT += 1
            f.write('t,t,'+seriesuid+','+str(c1)+','+str(c2)+','+str(c3)+',')
            print coord
            coords.pop(i)
            diameter.pop(i)
            ff = False
            break
        if not ff: break
      if not ff: break
    if not ff:
      correct.append(l)
      label *= 0
      label = np.where(labels==l,1,0)
      props = measure.regionprops(label,pseg)
      propsi = measure.regionprops(label,image)
      centroid = props[0].centroid
      centroid = np.round(np.asarray(centroid)).astype(int)
      cx, cy, cz = centroid
      mx_p = props[0].max_intensity
      mean_p = props[0].mean_intensity
      area = props[0].area
      bbox = props[0].bbox
      du = bbox[3]-bbox[0]
      dv = bbox[4]-bbox[1]
      dw = bbox[5]-bbox[2]
      bbox_area = du*dv*dw
      mx_i = propsi[0].max_intensity
      mn_i = propsi[0].min_intensity
      irange = abs(mn_i-mx_i)
      mean_i = propsi[0].mean_intensity
      f.write(str(cx)+','+str(cy)+','+str(cz)+','+str(area)+','+str(bbox_area)+','+str(mx_p)+','+str(mean_p)+','+str(mx_i)+','+str(mean_i)+'\n')
      props = str(area)+'-'+str(bbox_area)+'-'+str(mx_p)+'-'+str(mean_p)+'-'+str(mx_i)+'-'+str(mean_i)+'-'+str(irange)+'-'
      roi = image[cx-32:cx+32,cy-32:cy+32,cz-32:cz+32]
      if roi.shape != (64,64,64):
        print "ROI DEBUGGING"
        print roi.shape
        xl,xh,yl,yh,zl,zh = 0,0,0,0,0,0
        ishape = image.shape
        print ishape
        if roi.shape[0] == 0 or roi.shape[0] != 64:
          print "DIM 0"
          if cx-32 < 0:
            xl = abs(cx-32)
            print xl
          elif cx+32 > ishape[0]:
            xh = cx+32-ishape[0]+1
            print xh
          else:
            print "ERROR ROI RESHAPE X"
        if roi.shape[1] == 0 or roi.shape[1] != 64:
          if cy-32 < 0:
            yl = abs(cy-32)
          elif cy+32 > ishape[1]:
            yh = cy+32-ishape[1]+1
          else:
            print "ERROR ROI RESHAPE Y"
        if roi.shape[2] == 0 or roi.shape[2] != 64:
          print "DIM 2"
          if cz-32 < 0:
            zl = abs(cz-32)
            print zl
          elif cz+32 > ishape[2]:
            zh = cz+32-ishape[2]+1
            print zh
          else:
            print "ERROR ROI RESHAPE Z"
        roi = np.zeros((64,64,64))
        roi[xl:64-xh,yl:64-yh,zl:64-zh] = image[cx-32+xl:cx+32-xh,cy-32+yl:cy+32-yh,cz-32+zl:cz+32-zh]
        print roi.shape
      tmp = props
      props=str(COUNT)+'-'
      props+=tmp
      writeCand(roi,seriesuid,outpath,props,True)
  print "RANGE: " + str(np.max(labels))
  for n in range(1,np.max(labels)):  
    if (n+1)%100==0:
      print "CANDIDATE NO: "+str(n+1)
    if n not in correct:     
      label *= 0
      label = np.where(labels==n,1,0)
      props = measure.regionprops(label,pseg)
      propsi = measure.regionprops(label,image)
      centroid = props[0].centroid
      centroid = np.round(np.asarray(centroid)).astype(int)
      cx, cy, cz = centroid
      ff = True
      for i in range(len(coords)):
        coord = coords[i]
        coord = (coord[2],coord[1],coord[0])
        d = diameter[i]  
        if 0.75*d >= getDistance(np.asarray(props[0].centroid),coord):
          COUNT += 1
          f.write('t,t,'+seriesuid+','+str(coord[2])+','+str(coord[1])+','+str(coord[0])+',')
          print coord
          print centroid
          coords.pop(i)
          diameter.pop(i)
          ff = False
          break
      if ff:
        f.write('f,f,'+seriesuid+',0,0,0,')  
      mx_p = props[0].max_intensity
      mean_p = props[0].mean_intensity
      area = props[0].area
      bbox = props[0].bbox
      du = bbox[3]-bbox[0]
      dv = bbox[4]-bbox[1]
      dw = bbox[5]-bbox[2]
      bbox_area = du*dv*dw
      mx_i = propsi[0].max_intensity
      mn_i = propsi[0].min_intensity
      irange = abs(mn_i-mx_i)
      mean_i = propsi[0].mean_intensity
      f.write(str(cx)+','+str(cy)+','+str(cz)+','+str(area)+','+str(bbox_area)+','+str(mx_p)+','+str(mean_p)+','+str(mx_i)+','+str(mean_i)+'\n')
      props = str(area)+'-'+str(bbox_area)+'-'+str(mx_p)+'-'+str(mean_p)+'-'+str(mx_i)+'-'+str(mean_i)+'-'+str(irange)+'-'
      roi = image[cx-32:cx+32,cy-32:cy+32,cz-32:cz+32]
      if roi.shape != (64,64,64):
        print "ROI DEBUGGING"
        print roi.shape
        xl,xh,yl,yh,zl,zh = 0,0,0,0,0,0
        ishape = image.shape
        print ishape
        if roi.shape[0] == 0 or roi.shape[0] != 64:
          print "DIM 0"
          if cx-32 < 0:
            xl = abs(cx-32)
            print xl
          elif cx+32 > ishape[0]:
            xh = cx+32-ishape[0]+1
            print xh
          else:
            print "ERROR ROI RESHAPE X"
        if roi.shape[1] == 0 or roi.shape[1] != 64:
          if cy-32 < 0:
            yl = abs(cy-32)
          elif cy+32 > ishape[1]:
            yh = cy+32-ishape[1]+1
          else:
            print "ERROR ROI RESHAPE Y"
        if roi.shape[2] == 0 or roi.shape[2] != 64:
          print "DIM 2"
          if cz-32 < 0:
            zl = abs(cz-32)
            print zl
          elif cz+32 > ishape[2]:
            zh = cz+32-ishape[2]+1
            print zh
          else:
            print "ERROR ROI RESHAPE Z"
        roi = np.zeros((64,64,64))
        roi[xl:64-xh,yl:64-yh,zl:64-zh] = image[cx-32+xl:cx+32-xh,cy-32+yl:cy+32-yh,cz-32+zl:cz+32-zh]
        print roi.shape    
      if ff:
        FCOUNT += 1
        tmp = props
        props=str(FCOUNT)+'-'
        props+=tmp
        writeCand(roi,seriesuid,outpath,props,False)
      else:
        tmp = props
        props=str(COUNT)+'-'
        props+=tmp
        writeCand(roi,seriesuid,outpath,props,True)

  for i in range(len(coords)):
    f.write('t,f,'+seriesuid+',0,0,0,0,0,0,0,0,0,0,0,0\n')
  f.close()


def run():
  global MODEL
  global TEST
  global VAL
  global DATAPATH
  global WEIGHTSPATH
  global OUTPATH
  global OUTFILE
  CT = 0
  print "TEST: "+str(TEST)+"   VAL: "+str(VAL)+"   MODEL: "+str(MODEL)
  filelist, filepathlist = getFilelist(DATAPATH)
  model = seg3d.getModel()
  model.load_weights(WEIGHTSPATH)
  candlist = pd.read_csv(ANNOTATIONS)
  time1 = time.time()
  flag = False
  #for i in range(len(filelist)):
  #  print filelist[i]
  #return
  for n in range(START,len(filelist)):
    if scanids is not None:
      for scan in scanids:
        #print n
        flag = True
        if filelist[n] == scan:
          flag = False 
          break
    if flag:
      flag = False
      continue
    t1 = time.time()
    print "FILE NO " + str(n) + "   UID: " + filelist[n]
    currentNodules = candlist[candlist['seriesuid'] == filelist[n]]
    itk = sitk.ReadImage(filepathlist[n])
    image, _ = cg.loadItk(filepathlist[n],None)
    origin = np.array(itk.GetOrigin())
    spacing = np.array(itk.GetSpacing())
    coords, d = getCoords(currentNodules, origin, spacing)
    print coords
    #TODO: ADD TO OTHER SEGMENT_x_.PY
    if len(coords) == 0:
      continue
    CT += 1
    segmentation = cd.segmentLung(image,reseg=False)
    if cd.checkSeg(segmentation): segmentation = cd.segmentLung(image,reseg=True)
    mask = cd.applyMask(image, segmentation)
    segmentedImage = copy.deepcopy(image)
    segmentedImage[mask==0] = 0
    zmin,zmax,ymin,ymax,xmin,xmax = cd.findROI(segmentedImage)
    roi = cd.crop(segmentedImage)
    if roi.shape[0]<100 or roi.shape[1]<100 or roi.shape[2]<100:
      segmentation = cd.segmentLung(image, error=True)
      mask = cd.applyMask(image, segmentation)
      segmentedImage = copy.deepcopy(image)
      segmentedImage[mask==0] = 0
      zmin,zmax,ymin,ymax,xmin,xmax = cd.findROI(segmentedImage)
      roi = cd.crop(segmentedImage)
    shift = (xmin,ymin,zmin) #**XYZ NOT ZYX
    segShape = getSegmentationShape(roi.shape)#, sixteen=SIXTEEN)#image.shape)
    lpad, hpad = getPadding(segShape,roi)#image)
    segImage = np.zeros(segShape)
    segImage[lpad[0]:segShape[0]-hpad[0],lpad[1]:segShape[1]-hpad[1],lpad[2]:segShape[2]-hpad[2]] = roi#image 
    segImage = np.clip(segImage,-1000,1000)
    segImage += 1000
    segImage *= (255./2000.)
    print "SEGMENTING ..."
    stride = STRIDE
    strides = copy.deepcopy(segShape)
    strides /= 32
    print segShape
    print strides
    segmentation = segmentImage(segImage, model, strides, stride)
    #if stride == 32:
    lpad -= 16 
    hpad -= 16
    #elif stride == 16:
    #  lpad -= 8
    #  hpad -= 8
    #print segmentation.shape
    #print mask.shape
    #segmentation[mask==0] = 0
    tmpseg = segmentation[lpad[0]:segmentation.shape[0]-hpad[0],lpad[1]:segmentation.shape[1]-hpad[1],lpad[2]:segmentation.shape[2]-hpad[2]]
    #tmpseg[mask==0] = 0
    pseg = np.zeros(image.shape)
    pseg[zmin:zmax,ymin:ymax,xmin:xmax] = tmpseg
    seg = np.zeros(pseg.shape)
    seg = np.where(pseg>THRESH,1,0)
    #pseg[mask==0] = 0
    #seg[mask==0] = 0
    print "ASSESSING ..."
    labels = measure.label(seg, connectivity=1)
    if np.max(labels) >= MAXCANDS:
      print "TOO MANY: " + str(np.max(labels))
      continue
    assess(OUTFILE,labels,coords,filelist[n],OUTPATH,pseg,image,d)
    t2 = time.time()
    print "LAST SCAN TIME: "+str((t2-t1)/60)+'m '+str((t2-t1)%60)+'s'
  time2 = time.time()
  print "TOTAL TIME: "+str((time2-time1)/60)+'m '+str((time2-time1)%60)+'s'
  print "TOTAL SCANS PROCESSED: "+str(CT)


if __name__ == '__main__':
  global MODEL
  global TEST
  global VAL
  global DATAPATH
  global WEIGHTSPATH
  global OUTPATH
  global OUTFILE
  MODEL = int(sys.argv[3])
  VAL = int(sys.argv[2])
  TEST = int(sys.argv[1])
  DATAPATH =  '/hdfs/a/subset'+str(VAL)+'/'
  WEIGHTSPATH = '/hdfs/b/nodules/segmodels/'+str(TEST)+'/'+str(VAL)+'/'+str(MODEL)+'.h5'
  OUTPATH = '/hdfs/b/nodules/segmodels/'+str(TEST)+'/'+str(VAL)+'/'+str(MODEL)+'/'
  OUTFILE = '/hdfs/b/nodules/segmodels/'+str(TEST)+'/'+str(VAL)+'/'+str(MODEL)+'.txt'
  run()

