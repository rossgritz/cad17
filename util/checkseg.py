import subprocess
import time

def main():
  while True:
    fin1 = open('seglog.dat','r')
    fout1 = open('tmp.dat','w')
    cases = []
    for line in fin1:
      l = line.strip()
      if l[0] != '#':
        l = l.split()
        print l
        test = int(l[1])
        val = int(l[3]) 
        model = int(l[5])
        case = [test,val,model]
        cases.append(case)
        fout1.write("#"+line)
      else:
        fout1.write(line)
    fout1.close()
    fin1.close()  
    subprocess.call('mv tmp.dat seglog.dat',shell=True)

    for case in cases:
      fin2 = open('/home/g/nodules/segmodels/'+str(case[0])+'/'+str(case[1])+'/'+\
                   str(case[2])+'/status.txt','r')
      lossl = []
      lossd = {}
      for line in fin2:
        l = line.split()
        if l[0][0]!='T':
          if int(l[2]) == 1:
            lossl = []
            lossd = {}
        else:
          continue
        lossl.append(float(l[6].strip('--')))
        lossd[int(l[2])] = float(l[6].strip('--'))
      lossl.sort()
      losses = lossl[:10]
      cases_ = []
      for loss in losses:
        cases_.append(next(key for key, value in lossd.items() if value == loss))
      cases_.sort()
      command = 'cd ~/nodules/; python genscpg.py 0 '+str(case[0])+' '+str(case[1])+' '+str(case[2])+' '
      for case_ in cases_:
        command += str(case_)+' '
      print command
      subprocess.call([command],shell=True)
      fin2.close()
      
    time.sleep(60)

if __name__=='__main__':
  main()
