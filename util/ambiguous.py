import sys
sys.path.append('/home/rrg0013@auburn.edu/git/nodules/src/')

import lidc
import os 
import SimpleITK as sitk
import pandas as pd
import numpy as np
from collections import defaultdict
from glob import glob

BINNO = 0
LUNA_PATH = '/scr/data/nodules/luna/test/subset'+str(BINNO)+'/'
CAND_DAT_PATH = '/scr/data/nodules/testseg/final/finalt/'+BINNO_+'_.txt'
CAND_PATH = '/scr/data/nodules/testseg/final/finalt/clean/f/'+BINNO_+'/'
XMLDIR = '/home/rrg0013@auburn.edu/xmlc/'
OUT_PATH = '/scr/data/nodules/testseg/final/finalt/clean/t/'+BINNO_+'/'

print CAND_DAT_PATH
print CAND_PATH
print OUT_PATH

df = pd.read_csv(CAND_DAT_PATH,header=None)

cdf = df.iloc[:,[2,6,7,8,9,10,11,12,13,14]]
dat = cdf.as_matrix()

uids = df.iloc[:,2]
uids = set(uids)
uids = list(uids)
cands = {}
for uid in uids:
  tdf = df[df.iloc[:,2] == uid]
  tdf = tdf.iloc[:,6:15]
  cands[uid] = tdf
fnames = defaultdict(list)
files = glob(CAND_PATH+'/*.png')
for fpath in files:
  fname = fpath.split('-')
  fname = fname[-1]
  fname = fname[:-4]
  fnames[fname].append(fpath)


def run():
  ct = 0
  for parent, subdir, files in os.walk(LUNA_PATH):
    for fname in files:
      if ".mhd" in fname.lower():
        fpath = os.path.join(parent,fname)
        itk = sitk.ReadImage(fpath)
        origin = np.array(itk.GetOrigin())
        spacing = np.array(itk.GetSpacing())
        fname = fname[:-4]
        scan = lidc.xml(XMLDIR,fname,spacing,origin)
        lidc.cluster(scan)
        scan.combine()
        cand = cands[fname]
        fpaths = fnames[fname]
        tcoords = cand.iloc[:,0:3].as_matrix()
        coords = []
        for i in range(tcoords.shape[0]):
          coords.append((tcoords[i,2],tcoords[i,1],tcoords[i,0]))
        for nodule in scan.nodules:
          centroid = nodule.centroid
          d = nodule.diameter
          for coord in coords:
            dist = scan.getDistance(centroid,coord)
            if d*.75 > dist:
              print "found"
              tdf = cand[cand.iloc[:,0]==coord[2]] 
              tdf = tdf[tdf.iloc[:,1]==coord[1]]
              tdf = tdf[tdf.iloc[:,2]==coord[0]]
              print tdf
              tm = tdf.as_matrix()
              cx,cy,cz,ca,cb,cc,cd,ce,cf = tm[0,0],tm[0,1],tm[0,2],tm[0,3],tm[0,4],tm[0,5],tm[0,6],tm[0,7],tm[0,8]
              for fpath in fpaths:
                fvals = fpath.split('_')
                fvals = fvals[-1]
                fvals = fvals.split('-')
                try:
                  no_,a,b,c,d,e,f,_,__ = fvals
                except ValueError:
                  try:
                    no_,a,b,c,d,_,e,__,f,___,____ = fvals
                  except ValueError:
                    no_,a,b,c,d,e,_,f,__,___ = fvals
                if int(float(ca)) == int(float(a)):
                  if int(float(cb)) == int(float(b)):
                    if int(float(cc)) == int(float(c)):
                      print "OLD FILEPATH"
                      print fpath
                      ct += 1
                      print ct
                      npath = OUT_PATH + '/t_b_' + str(ct) + '_' + str(BINNO) + '.png'
                      print npath
                      try:
                        print "MOVING ..."
                        os.rename(fpath,npath)
                      except:
                        print "\n\n\nERROR OS\n\n\n"
                    else:  
                      print "\n\n\nERROR D\n\n\n"
              coords.remove(coord)
              break
  print "TOTAL " + str(ct)
  return ct

ct = run()

print ct



