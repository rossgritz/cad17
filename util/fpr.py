import pandas as pd 
import numpy as np
from glob import glob
import os

THRESH = 7

fps = ['/hdfs/a/nodules/segmodels/6/7/101/'\
     ,'/hdfs/a/nodules/segmodels/2/3/107/'\
     ,'/hdfs/a/nodules/segmodels/2/3/35/'\
     ,'/hdfs/a/nodules/segmodels/1/2/25/'\
     ,'/hdfs/a/nodules/segmodels/1/3/43/'\
     ,'/hdfs/a/nodules/segmodels/1/5/36/'\
     ,'/hdfs/a/nodules/segmodels/2/3/35/'\
     ,'/hdfs/a/nodules/segmodels/2/0/41/'\
     ,'/hdfs/a/nodules/segmodels/2/3/27/'\
     ,'/hdfs/a/nodules/segmodels/2/7/40/'\
     ,'/hdfs/a/nodules/segmodels/3/0/29/'\
     ,'/hdfs/a/nodules/segmodels/3/1/36/'\
     ,'/hdfs/a/nodules/segmodels/3/7/35/'\
     ,'/hdfs/a/nodules/segmodels/3/9/37/'\
     ,'/hdfs/a/nodules/segmodels/2/6/41/']

nps = ['/hdfs/a/nodules/segmodels/6/7/101c/'\
     ,'/hdfs/a/nodules/segmodels/2/3/107c/'\
     ,'/hdfs/a/nodules/segmodels/2/3/35c/'\
     ,'/hdfs/a/nodules/segmodels/1/2/25c/'\
     ,'/hdfs/a/nodules/segmodels/1/3/43c/'\
     ,'/hdfs/a/nodules/segmodels/1/5/36c/'\
     ,'/hdfs/a/nodules/segmodels/2/3/35c/'\
     ,'/hdfs/a/nodules/segmodels/2/0/41c/'\
     ,'/hdfs/a/nodules/segmodels/2/3/27c/'\
     ,'/hdfs/a/nodules/segmodels/2/7/40c/'\
     ,'/hdfs/a/nodules/segmodels/3/0/29c/'\
     ,'/hdfs/a/nodules/segmodels/3/1/36c/'\
     ,'/hdfs/a/nodules/segmodels/3/7/35c/'\
     ,'/hdfs/a/nodules/segmodels/3/9/37c/'\
     ,'/hdfs/a/nodules/segmodels/2/6/41c/']



ct = 0
for binno in range(0,len(fps)):
  fp = fps[binno]
  np = nps[binno]
  print np
  if not os.path.exists(np): os.makedirs(np)
  flist = glob(fp+'*.png')
  #print flist
  for f in flist:
    fdat = f.split('_')
    fn = f.split('/')
    fn = fn[-1]
    tdat = fdat[-2]
    #print tdat
    if tdat[-1] == 't' or tdat[-1] == 'f':
      fdat = fdat[-1]
      area = fdat.split('-')
      area = area[1]
      area = int(area)
      if area >= THRESH:
        ct+=1
        npath = np+fn
        #print area
        #print npath
        os.rename(f,npath)
   

