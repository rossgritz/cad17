import pandas as pd 
import numpy as np
from glob import glob
import os


fps = ['/scr/data/nodules/testseg/final/finalt/zero_/'\
     ,'/scr/data/nodules/testseg/final/final/one/'\
     ,'/scr/data/nodules/testseg/final/final/two/'\
     ,'/scr/data/nodules/testseg/final/final/three/'\
     ,'/scr/data/nodules/testseg/final/final/four/'\
     ,'/scr/data/nodules/testseg/final/final/five/'\
     ,'/scr/data/nodules/testseg/final/final/six/'\
     ,'/scr/data/nodules/testseg/final/final/seven/'\
     ,'/scr/data/nodules/testseg/final/final/eight/'\
     ,'/scr/data/nodules/testseg/final/final/nine/']

nps = ['/scr/data/nodules/testseg/final/finalt/clean/t/zero/'\
     ,'/scr/data/nodules/testseg/final/final/clean/t/one/'\
     ,'/scr/data/nodules/testseg/final/final/clean/t/two/'\
     ,'/scr/data/nodules/testseg/final/final/clean/t/three/'\
     ,'/scr/data/nodules/testseg/final/final/clean/t/four/'\
     ,'/scr/data/nodules/testseg/final/final/clean/t/five/'\
     ,'/scr/data/nodules/testseg/final/final/clean/t/six/'\
     ,'/scr/data/nodules/testseg/final/final/clean/t/seven/'\
     ,'/scr/data/nodules/testseg/final/final/clean/t/eight/'\
     ,'/scr/data/nodules/testseg/final/final/clean/t/nine/']


ct = 0
for binno in range(0,1):
  fp = fps[binno]
  np = nps[binno]
  flist = glob(fp+'*.png')
  #print flist
  for f in flist:
    fdat = f.split('_')
    fn = f.split('/')
    fn = fn[-1]
    tdat = fdat[-2]
    #print tdat
    if tdat[-1] == 't':
      fdat = fdat[-1]
      area = fdat.split('-')
      area = area[1]
      area = int(area)
      if area > 7:
        ct+=1
        npath = np+fn
        #print area
        #print npath
        os.rename(f,npath)
   

