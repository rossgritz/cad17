import time
import os
from glob import glob
import subprocess


def genSubdirs(fp):
  fp = fp.strip()
  fps = fp.split('/')
  val = fps[-1]
  test = fps[-2]
  i = 0
  while os.path.exists(fp+'/'+str(i)+'/'): i+=1
  nfp = fp+'/'+str(i)+'/'
  os.makedirs(nfp)
  fin = open(fp+'/test.'+test+'.'+val+'.sh','r')
  fout = open(fp+'/tmp.sh','w')
  for line in fin:
    if line[0:4] == 'grep':
      #print line
      l = line.strip()
      l = l.split('/')
      #print l
      newline = l[0]+'/'+l[1]+'/'+l[2]+'/'+str(i)+'/'+l[3]
      #print newline
      fout.write(newline+'\n')
    else:
      fout.write(line)
  fin.close()
  fout.close()
  command = 'mv '+fp+'/tmp.sh '+nfp+'/test.'+str(test)+'.'+str(val)+'.sh'
  print command
  subprocess.call([command],shell=True)
  command = 'rm '+fp+'/test.'+test+'.'+val+'.sh'
  print command
  subprocess.call([command],shell=True)
  command = 'mv '+fp+'/*tgz '+nfp
  print command
  subprocess.call([command],shell=True)
  command = 'mv '+fp+'/*.h5 '+nfp
  print command 
  subprocess.call([command],shell=True)
  fin = open(fp+'/segcheck.'+test+'.'+val+'.sh','r')
  fout = open(fp+'/tmp.sh','w')
  for line in fin:
    if line[0:4] == 'CUDA':
      l = line.strip()
      l = l.split()
      try:
        newline = l[0]+' '+l[1]+' '+l[2]+' '+l[3]+' '+l[4]+' '+str(i)+' '+l[5]+' '+l[6]
      except IndexError:  
        newline = l[0]+' '+l[1]+' '+l[2]+' '+l[3]+' '+l[4]+' '+str(i)+' '+l[5]
      #print newline
      fout.write(newline+'\n')
    else:
      #print line
      fout.write(line)
  fin.close()
  fout.close()
  command = 'mv '+fp+'/tmp.sh '+nfp+'/segcheck.'+test+'.'+val+'.sh '
  print command
  subprocess.call([command],shell=True)
  command = 'rm '+fp+'/segcheck.'+test+'.'+val+'.sh'
  print command
  subprocess.call([command],shell=True)
  return str(i), test, val


def main():
  while True:
    newfiles = [y for x in os.walk('.') for y in glob(os.path.join(x[0], 'segcheck*.sh'))]
    for f in newfiles:
      fp = os.path.dirname(f)
      sd, test, val = genSubdirs(fp)
      tarballs = glob(fp+'/'+str(sd)+'/*tgz')
      for tball in tarballs:
        command = "tar -xzf "+str(tball)+" -C "+str(fp+'/'+sd+'/')
        print command
        subprocess.call([command],shell=True)
        command = "rm "+str(tball)
        print command
        subprocess.call([command],shell=True)
      fname = os.path.basename(f)
      command = "mv "+fp+"/"+sd+"/"+fname+" ~/git/nodules/segproc/queue/segcheck."+test+"."+val+"."+sd+".sh"
      print command
      subprocess.call([command],shell=True)
      print "SENT TO QUEUE: "+str(f)
    time.sleep(60)
  return 0


if __name__ == '__main__':
  main()
