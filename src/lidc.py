import scan as scn
import nodule as nd
import candgen as cg
import pandas as pd
import numpy as np
import xml.etree.ElementTree as et


def cluster(scan):
  print scan.id
  nodules = []
  gno = 0

  for i, nodule in enumerate(scan.nodules):
    if nodule.type == 'L':
      nodule.no = i
      nodules.append(nodule)

  for i in range(len(nodules)):
    if not nodules[i].grouped:
      group = []
      group.append(i)
      for j in range(len(nodules)):
        if i != j:
          if not nodules[j].grouped:
            distance = scan.getDistance(nodules[i].centroid,nodules[j].centroid)
            if distance < (nodules[i].diameter/2.+nodules[j].diameter/2.):
              group.append(j)
      for j in range(1,len(group)):
        j = group[j]
        for k in range(len(nodules)):
          if i != k and j != k and i != j:
            if not nodules[k].grouped:
              distance = scan.getDistance(nodules[j].centroid,nodules[k].centroid)
              if distance < (nodules[j].diameter/2.+nodules[k].diameter/2.):
                group.append(k)
      if len(group) >= 1:
        group = set(group)
        group = list(group)
        for n in range(len(group)):
          nodules[group[n]].addGroup(gno,group)
        scan.groups.append(group)
        gno += 1


def xml(XMLDIR, seriesuid, spacing=None, origin=None):
  print XMLDIR
  scan = scn.Scan()
  scan.id = seriesuid
  xmlpath = XMLDIR+'/'+seriesuid+'.xml'
  print xmlpath
  try:
    xml = et.parse(xmlpath)
  except:
    return None
  ct = 0
  for reader in xml.findall('./{http://www.nih.gov}readingSession'):
    for reading in reader.findall('./{http://www.nih.gov}unblindedReadNodule'):
      nn = nd.Nodule() 
      if reading.find('./{http://www.nih.gov}characteristics') is not None:
        nn.type = 'L'
        characteristics = reading.find('./{http://www.nih.gov}characteristics')
        nn.addCharacteristics(characteristics)
        for roi in reading.findall('./{http://www.nih.gov}roi'):
          z = roi.find('./{http://www.nih.gov}imageZposition')
          z = float(z.text)
          z = z-origin[2]
          oz = z/spacing[2]
          include = roi.find('./{http://www.nih.gov}inclusion')
          if include.text == 'TRUE':
            for e in roi.findall('./{http://www.nih.gov}edgeMap'):
              x = e.find('./{http://www.nih.gov}xCoord')
              y = e.find('./{http://www.nih.gov}yCoord')
              coord = int(x.text),int(y.text)
              coord = (coord[0],coord[1],0.)
              coord *= spacing
              coord[2] = z
              nn.roi.append(coord)
              nn.oroi.append((int(x.text),int(y.text),oz))
        nn.getProperties()
        scan.nodules.append(nn)
      #else:
      #  nn.type = 'S'
      #  nn.diameter = 3.
      #  roi = reading.find('./{http://www.nih.gov}roi')
      #  e = roi.find('./{http://www.nih.gov}edgeMap')
      #  x = e.find('./{http://www.nih.gov}xCoord')
      #  x = float(x.text)
      #  x *= spacing[0]
      #  y = e.find('./{http://www.nih.gov}yCoord')
      #  y = float(y.text)
      #  y *= spacing[1]
      #  z = roi.find('./{http://www.nih.gov}imageZposition')
      #  z = float(z.text)
      #  z = z-origin[2]
      #  nn.ox = x
      #  nn.oy = y
      #  nn.oz = z
      #  nn.centroid = (x,y,z)
      #  scan.nodules.append(nn)
      
  return scan













