#import dicom
import os
#import math
#import time
import xml.etree.ElementTree as et
#import skimage.measure as measure
#import scipy.ndimage
#import numpy as np
#import pandas as pd



def collectXML(path,outpath):
  for parent, subdir, files in os.walk(path):
    for filename in files:
      if ".xml" in filename.lower():
        xmlpath = os.path.join(parent, filename)
        xml = et.parse(xmlpath)
        header = xml.find('./{http://www.nih.gov}ResponseHeader')
        if header is not None:
          seriesuid = header.find('./{http://www.nih.gov}SeriesInstanceUid')
          #if seriesuid.text == '1.3.6.1.4.1.14519.5.2.1.6279.6001.116097642684124305074876564522':
          #  print "FOUND"
          #  return
          newfilename = seriesuid.text + '.xml'
          newpath = os.path.join(outpath, newfilename)
          os.rename(xmlpath,newpath)
path = '/scr/data/nodules/unprocessed/'
newpath = '/scr/data/nodules/xmlc/'
collectXML(path,newpath)
