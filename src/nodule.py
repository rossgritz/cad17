import numpy as np
import skimage.measure as measure


class Nodule():
  def __init__(self):
    self.type = ''
    self.no = -1
    
    self.subtlety = 0.
    self.internalStructure = 0.
    self.sphericity = 0.
    self.calcification = 0.
    self.margin = 0.
    self.lobulation = 0.
    self.spiculation = 0.
    self.texture = 0.
    self.malignancy = 0.
    
    #original x, y, z and roi coords
    self.ox = 0.
    self.oy = 0. 
    self.oz = 0.
    self.centroid = None
    self.roi = []
    self.oroi = []
    
    self.group = -1
    self.grouped = False
    self.groupMembers = []

    self.tagCount = -1
    self.valid = False
    
    
  def addCharacteristics(self, chars, type=None):
    try:
      subtlety = chars.find("{http://www.nih.gov}subtlety")
      self.subtlety = int(subtlety.text)
      internal = chars.find("{http://www.nih.gov}internalStructure")
      self.internalStructure = int(internal.text)
      calcification = chars.find("{http://www.nih.gov}calcification")
      self.calcification = int(calcification.text)
      sphericity = chars.find("{http://www.nih.gov}sphericity")
      self.sphericity = int(sphericity.text)
      margin = chars.find("{http://www.nih.gov}margin")
      self.margin = int(margin.text)
      lobulation = chars.find("{http://www.nih.gov}lobulation")
      self.lobulation = int(lobulation.text)
      spiculation = chars.find("{http://www.nih.gov}spiculation")
      self.spiculation = int(spiculation.text)
      texture = chars.find("{http://www.nih.gov}texture")
      self.texture = int(texture.text)
      ##-MALIGNANCY IS ASSUMING 60 YEAR OLD MALE SMOKER ..
      malignancy = chars.find("{http://www.nih.gov}malignancy")
      self.malignancy = int(malignancy.text)
    except AttributeError:
      print "ATTRIBUTE ERROR"
        
  def addGroup(self,groupNo,group):
    self.group = groupNo
    self.grouped = True
    for i in range(len(group)):
      if group[i] != self.no:
        self.groupMembers.append(group[i])
      
  def getProperties(self):
    tmp = np.zeros((500,500,500)).astype(int)
    xmax,xmin,ymax,ymin,zmax,zmin=-1e5,1e5,-1e5,1e5,-1e5,1e5
    x,y,z=0,0,0
    for value in self.roi:
      xmax = max(abs(int(round(value[0]))),xmax)
      xmin = min(abs(int(value[0])),xmin)
      ymax = max(abs(int(round(value[1]))),ymax)
      ymin = min(abs(int(value[1])),ymin)
      zmax = max(abs(int(round(value[2]))),zmax)
      zmin = min(abs(int(value[2])),zmin)
      x,y,z = value[0],value[1],value[2]
      tmp[abs(int(x)),abs(int(y)),abs(int(z))]=1  

    props = measure.regionprops(tmp)
    centroid = np.array(props[0].centroid)
    self.centroid = centroid
    
    tmp = tmp[xmin-1:xmax+1,ymin-1:ymax+1,zmin-1:zmax+1]
    tmp[tmp>0] = 1
    if tmp.shape[0]*tmp.shape[1]*tmp.shape[2] == max(tmp.shape):
      self.diameter = 3.
      self.area = max(tmp.shape)
    else:
      props = measure.regionprops(tmp)
      self.diameter = props[0].equivalent_diameter
      self.area = props[0].area
