import sys

def main():
  if len(sys.argv) < 5:
    return

  fout = open('segmodels/'+str(sys.argv[2])+'/'+str(sys.argv[3])+'/segcheck'+str(sys.argv[4])+'gen.sh','w')
  fout.write("#!/bin/bash\n")
  for i in range(5,len(sys.argv)):
    fout.write('CUDA_VISIBLE_DEVICES='+sys.argv[1]+' python src/segment0.py '+str(sys.argv[2])+\
               ' '+str(sys.argv[3])+' '+str(sys.argv[i])+' ')
    if i != len(sys.argv) - 1:
      fout.write('&&\n')
    else:
      fout.write('\n')
  fout.close()


  fout = open('scp.'+str(sys.argv[2])+'.'+str(sys.argv[3])+'.sh','w')
  fout.write('#!/bin/bash\n')
  for i in range(5,len(sys.argv)):
    fout.write('scp segmodels/'+str(sys.argv[2])+'/'+str(sys.argv[3])+'/'+str(sys.argv[4])+'/'+str(sys.argv[i])+'.h5'+\
               ' kristof@192.168.0.109:/Users/krsitof/Desktop/ntmp/'+str(sys.argv[2])+'/'+str(sys.argv[3])+'/\n')
  fout.write('chmod +x segmodels/'+str(sys.argv[2])+'/'+str(sys.argv[3])+'/segcheck'+str(sys.argv[4])+'gen.sh\n')
  fout.write('scp segmodels/'+str(sys.argv[2])+'/'+str(sys.argv[3])+'/segcheck'+str(sys.argv[4])+'gen.sh'+\
             ' kristof@192.168.0.109:/Users/kristof/Desktop/ntmp/'+str(sys.argv[2])+'/'+str(sys.argv[3])+'/\n')
  fout.write('chmod +x segmodels/'+str(sys.argv[2])+'/'+str(sys.argv[3])+'/test.'+str(sys.argv[2])+'.'+str(sys.argv[3])+'.sh\n')
  fout.write('scp segmodels/'+str(sys.argv[2])+'/'+str(sys.argv[3])+'/test.'+str(sys.argv[2])+'.'+str(sys.argv[3])+'.sh'+\
             ' kristof@192.168.0.109:/Users/kristof/Desktop/ntmp/'+str(sys.argv[2])+'/'+str(sys.argv[3])+'/\n')
  fout.write('scp segmodels/'+str(sys.argv[2])+'/'+str(sys.argv[3])+'/scpa.'+str(sys.argv[2])+'.'+str(sys.argv[3])+'.sh'+\
             ' kristof@192.168.0.109:/Users/kristof/Desktop/ntmp/'+str(sys.argv[2])+'/'+str(sys.argv[3])+'/\n')
  fout.write('scp segmodels/'+str(sys.argv[2])+'/'+str(sys.argv[3])+'/scpb.'+str(sys.argv[2])+'.'+str(sys.argv[3])+'.sh'+\
             ' kristof@192.168.0.109:/Users/kristof/Desktop/ntmp/'+str(sys.argv[2])+'/'+str(sys.argv[3])+'/\n')
  fout.close()


  fout = open('scpa.'+str(sys.argv[2])+'.'+str(sys.argv[3])+'.sh','w')
  fout.write('#!/bin/bash\n')
  for i in range(5,len(sys.argv)):
    fout.write('scp '+str(sys.argv[2])+'/'+str(sys.argv[3])+'/'+str(sys.argv[4])+'/'+str(sys.argv[i])+'.h5'+\
               ' admin@131.204.65.198:/home/admin/ntmp/'+str(sys.argv[2])+'/'+str(sys.argv[3])+'/\n')
  fout.write('chmod +x segmodels/'+str(sys.argv[2])+'/'+str(sys.argv[3])+'/segcheck'+str(sys.argv[4])+'gen.sh\n')
  fout.write('scp '+str(sys.argv[2])+'/'+str(sys.argv[3])+'/segcheck'+str(sys.argv[4])+'gen.sh'+\
             ' admin@131.204.65.198:/home/admin/ntmp/'+str(sys.argv[2])+'/'+str(sys.argv[3])+'/\n')
  fout.write('chmod +x segmodels/'+str(sys.argv[2])+'/'+str(sys.argv[3])+'/test.'+str(sys.argv[2])+'.'+str(sys.argv[3])+'.sh\n')
  fout.write('scp '+str(sys.argv[2])+'/'+str(sys.argv[3])+'/test.'+str(sys.argv[2])+'.'+str(sys.argv[3])+'.sh'+\
             ' admin@131.204.65.198:/home/admin/ntmp/'+str(sys.argv[2])+'/'+str(sys.argv[3])+'/\n')
  fout.write('scp '+str(sys.argv[2])+'/'+str(sys.argv[3])+'/scpb.'+str(sys.argv[2])+'.'+str(sys.argv[3])+'.sh'+\
             ' admin@131.204.65.198:/home/admin/ntmp/'+str(sys.argv[2])+'/'+str(sys.argv[3])+'/\n')
  fout.close()


  fout = open('scpb.'+str(sys.argv[2])+'.'+str(sys.argv[3])+'.sh','w')
  fout.write('#!/bin/bash\n')
  for i in range(5,len(sys.argv)):
    fout.write('scp '+str(sys.argv[2])+'/'+str(sys.argv[3])+'/'+str(sys.argv[4])+'/'+str(sys.argv[i])+'.h5'+\
               ' node2:/hdfs/b/nodules/segmodels/'+str(sys.argv[2])+'/'+str(sys.argv[3])+'/\n')
  fout.write('chmod +x segmodels/'+str(sys.argv[2])+'/'+str(sys.argv[3])+'/segcheck'+str(sys.argv[4])+'gen.sh\n')
  fout.write('scp '+str(sys.argv[2])+'/'+str(sys.argv[3])+'/segcheck'+str(sys.argv[4])+'gen.sh'+\
             ' node2:/hdfs/b/nodules/segmodels/'+str(sys.argv[2])+'/'+str(sys.argv[3])+'/\n')
  fout.write('chmod +x segmodels/'+str(sys.argv[2])+'/'+str(sys.argv[3])+'/test.'+str(sys.argv[2])+'.'+str(sys.argv[3])+'.sh\n')
  fout.write('scp '+str(sys.argv[2])+'/'+str(sys.argv[3])+'/test.'+str(sys.argv[2])+'.'+str(sys.argv[3])+'.sh'+\
             ' node2:/hdfs/b/nodules/segmodels/'+str(sys.argv[2])+'/'+str(sys.argv[3])+'/\n')
  fout.close()


  fout = open('segmodels/'+str(sys.argv[2])+'/'+str(sys.argv[3])+'/test.'+str(sys.argv[2])+'.'+str(sys.argv[3])+'.sh','w')
  fout.write('#!/bin/bash\n')
  for i in range(5,len(sys.argv)):
    fout.write('\necho '+str(sys.argv[i])+':\n')
    fout.write('grep \'t,f\' segmodels/'+str(sys.argv[2])+'/'+str(sys.argv[3])+'/'+str(sys.argv[i])+'.txt | wc -l\n')
    fout.write('grep \'f,f\' segmodels/'+str(sys.argv[2])+'/'+str(sys.argv[3])+'/'+str(sys.argv[i])+'.txt | wc -l\n')
  fout.close()

if __name__ == '__main__':
  main()
