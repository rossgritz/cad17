##
# Remove manually identified images
# Images removed for remove directories for each cv bin
##

import os 

files = os.listdir('/scr/nodules/no4/bin9/remove9/')
removeFile = []
for file in files:
  file = file.strip()
  file = file.split('_')
  file = file[1]+'_'+file[2]
  file = file.split('.')
  file = file[:-1]
  nfile = ''
  for f in file:
    nfile = nfile + '.' + f
  nfile = nfile[1:-1]
  removeFile.append(nfile)
files = os.listdir('/scr/nodules/no4/bin9/')
for file in files:
  tag = file.strip()
  tag = tag.split('_')
  try:
    tag = tag[1] + '_' + tag[2]
  except IndexError:
    print tag
    continue
  tag = tag[0:-4]
  for i in range(len(removeFile)):
    if tag == removeFile[i]:
      oldpath = '/scr/nodules/no4/bin9/' + file
      newpath = '/scr/nodules/no4/bin9/remove9/' + file
      os.rename(oldpath,newpath)