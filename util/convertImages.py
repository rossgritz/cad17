import scipy.misc
from glob import glob 
import numpy as np
from matplotlib.pyplot import imsave

images = glob('*.png')
for img in images:
  image = scipy.misc.imread(img)
  newImage = np.zeros((128,256))
  #ct = 0
  for i in range(2,6):
    for j in range(8):
      #print "iteration "+str(ct)
      #ct+=1 
      #print "i "+str(i)+"    j "+str(j)
      newImage[(i-2)*32:((i-2)+1)*32,j*32:(j+1)*32]=image[i*64+16:(i+1)*64-16,j*64+16:(j+1)*64-16]
  scipy.misc.toimage(newImage,cmin=0,cmax=255).save(img)

