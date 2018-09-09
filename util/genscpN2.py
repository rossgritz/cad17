import sys

def main():
  if len(sys.argv) < 4:
    return

  fout = open('segmodels/'+str(sys.argv[2])+'/'+str(sys.argv[3])+'/segcheck'+str(sys.argv[1])+'gen.sh','w')
  fout.write("#!/bin/bash\n")
  for i in range(4,len(sys.argv)):
    fout.write('CUDA_VISIBLE_DEVICES='+sys.argv[1]+' python src/segment0.py '+str(sys.argv[2])+\
               ' '+str(sys.argv[3])+' '+str(sys.argv[i])+' ')
    if i != len(sys.argv)-1:
      fout.write('&&\n')
    else:
      fout.write('\n')
  fout.close()

  fout = open('scp.'+str(sys.argv[2])+'.'+str(sys.argv[3])+'.sh','w')
  fout.write('#!/bin/bash\n')
  for i in range(4,len(sys.argv)):
    fout.write('cp segmodels/'+str(sys.argv[2])+'/'+str(sys.argv[3])+'/0/'+str(sys.argv[i])+'.h5'+\
               ' /hdfs/b/nodules/segmodels/'+str(sys.argv[2])+'/'+str(sys.argv[3])+'/\n')
  fout.close()

  fout = open('segmodels/'+str(sys.argv[2])+'/'+str(sys.argv[3])+'/test.'+str(sys.argv[2])+'.'+str(sys.argv[3])+'.sh','w')
  fout.write('#!/bin/bash\n')
  for i in range(4,len(sys.argv)):
    fout.write('\necho '+str(sys.argv[i])+'\n')
    fout.write('grep \'t,f\' segmodels/'+str(sys.argv[2])+'/'+str(sys.argv[3])+'/'+str(sys.argv[i])+'.txt | wc -l\n')
    fout.write('grep \'f,f\' segmodels/'+str(sys.argv[2])+'/'+str(sys.argv[3])+'/'+str(sys.argv[i])+'.txt | wc -l\n')
  fout.close()

if __name__ == '__main__':
  main()
