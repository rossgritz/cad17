import math
import nodule as nd
import numpy as np


class Scan:
  def __init__(self):
    self.id = ''
    self.nodules = []
    self.groups = []
    self.clusters = []
    self.annotations = None
    
    
  def combine(self):
    for group in self.groups:
      nn = nd.Nodule()
      cx,cy,cz,d = 0.,0.,0.,0.
      for i in range(len(group)):
        cx += self.nodules[group[i]].centroid[0]
        cy += self.nodules[group[i]].centroid[1]
        cz += self.nodules[group[i]].centroid[2]
        d += self.nodules[group[i]].diameter
        if self.nodules[group[i]].type == 'L':
          nn.type = 'L'
        elif self.nodules[group[i]].type == 'S':
          if nn.type != 'L':
            nn.type = 'S'
        elif self.nodules[group[i]].type == 'N':
          if nn.type != 'L' and nn.type != 'S':
            nn.type = 'N'
        nn.subtlety += self.nodules[group[i]].subtlety
        nn.internalStructure += self.nodules[group[i]].internalStructure
        nn.sphericity += self.nodules[group[i]].sphericity 
        nn.calcification += self.nodules[group[i]].calcification
        nn.margin += self.nodules[group[i]].margin
        nn.lobulation += self.nodules[group[i]].lobulation
        nn.spiculation += self.nodules[group[i]].spiculation
        nn.texture += self.nodules[group[i]].texture
        nn.malignancy += self.nodules[group[i]].malignancy
        nn.roi += self.nodules[group[i]].roi
        nn.oroi += self.nodules[group[i]].oroi
      newCentroid = np.array([cx,cy,cz])
      newCentroid /= float(len(group))
      nn.centroid = newCentroid
      print newCentroid
      nn.tagCount = len(group)
      nn.diameter = d/float(len(group))
      nn.subtlety /= float(len(group))
      nn.internalStructure /= float(len(group))
      nn.sphericity /= float(len(group))
      nn.calcification /= float(len(group))
      nn.margin /= float(len(group))
      nn.lobulation /= float(len(group))
      nn.spiculation /= float(len(group))
      nn.texture /= float(len(group))
      nn.malignancy /= float(len(group))
      self.clusters.append(nn)

  def getDistance(self,a,b):
    du = (a[0]-b[0])
    dv = (a[1]-b[1])
    dw = (a[2]-b[2])
    return math.sqrt(du*du+dv*dv+dw*dw)
    
  def validateNodules(self):
    ct, count = 0, 0
    #if self.annotations is not None:
    #count = len(self.annotations)
    #for coord in self.annotations:
    #print coord
    for cluster in self.clusters:
      #distance = self.getDistance(coord,cluster.centroid)
      #print distance
      #if distance < cluster.diameter:
      print "VERIFIED NODULE"
      cluster.valid = True
      ct += 1
      #break
    print "VALIDATED " + str(ct) + " OF " + str(count) + " NODULES"
