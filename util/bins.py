##
# Data preprocessing - sorting
# Sorts raw scans into bins for cross validation
##


import os

binpath = './resources/bins/'
fpath = '/scr/nodules/LIDC-IDRI/'
newbinpath = '/scr/nodules/lidc-idri/'


def unique(line, c, i): 
  x = line.split(c) 
  return x[i]


def move(old, bin):
  new = newbinpath + '/bin' + str(bin) + '/' + unique(old,'/',6) + '/'
  if not os.path.exists(new): os.makedirs(new)
  try:
    os.rename(old, new)
  except OSError:
    new = newbinpath + '/bin' + str(bin) + '/' + unique(old,'/',6) + '_/'
  print new


def getbins():
  bins = []
  for parent, subdir, files in os.walk(binpath):
    for fname in files:
      bin = []
      fname = os.path.join(parent,fname)
      f = open(fname,'r')
      for line in f:
        try:
          bin.append(int(unique(line,'.',12)))
        except IndexError:
          continue
      f.close()
      bins.append(bin)
  return bins


def match(bins):
  for parent, subdir, files in os.walk(fpath):
    try:
      for i in range(len(bins)):
        print parent
        print unique(unique(parent,'/',5),'.',12)
        print i
        bin_ = bins[i]
        bin_ = [str(no) for no in bin_]
        print bin_
        if str(unique(unique(parent,'/',6),'.',12)) in bin_:
          old = parent 
          move(old, i)
    except IndexError:
      print 'INDEX ERROR'
      pass


def main():
  bins = getbins()
  match(bins)


if __name__ == '__main__':
  main()
