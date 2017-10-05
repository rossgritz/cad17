##
# Framework for running cases on multiple GPUs
#
##

import sys

sys.path.append('/home/rrg0013@auburn.edu/git/nodules/keras/')

import generatorModel as gen

import threading
import subprocess32 as sub
import os
import time
import random
import math
import json


OUTPATH = '/scr/data/nodules/init3dConv/'
COMMAND = ['python','keras/generatorModel.py']
GPU0 = False
GPU1 = False
GPU2 = False
GPU3 = False


class thread(threading.Thread):
  def __init__(self, gpu, data, train, val, init, date):#, args):
    threading.Thread.__init__(self)
    config = open('config.json').read()
    config = json.loads(config)
    self.gpu = gpu
    self.data = data
    self.train = train
    self.val = val
    self.init = str(init)
    self.date = config['date']
    #self.args = args
  def run(self):
    runCases(self.gpu,self.data,self.train,self.val,self.init,self.date)



def runCases(gpu, data, train, val, init, date):
  #fname = str(gpu) + '_' + str(init) + '_' + str(date) + '.out'
  #f = open(OUTPATH + '/' + date + '/' + fname,'a+')
  command = COMMAND + [data, train, val, init, date]
  os.environ['CUDA_VISIBLE_DEVICES']=str(gpu)
  #output = sub.check_output(command, stderr=sub.STDOUT, bufsize=1)
  #f.write(output)
  p = sub.Popen(command, stderr=sub.STDOUT, bufsize=1)
  #for line in iter(p.stdout.readline, b''):
  #  f.write(line)
  #p.stdout.close()
  p.wait()
  print 'exiting gpu thread ' + str(gpu)


def run(opts):
  global GPU0, GPU1, GPU2, GPU3 #,MAX_ITERATIONS
  opts.gpu = opts.gpu.split(',')
  if '3' in opts.gpu:
    GPU3=True
  if '2' in opts.gpu:
    GPU2=True
  if '1' in opts.gpu:
    GPU1=True
  if '0' in opts.gpu:
    GPU0=True
  data = opts.data
  train = opts.train
  val = opts.val
  init = int(opts.init)
  date = opts.date
  if GPU0 == True:
    t0 = thread(0,data,train,val,init,date)
    init += 100
  if GPU1 == True:
    t1 = thread(1,data,train,val,init,date)
    init += 100
  if GPU2 == True:
    t2 = thread(2,data,train,val,init,date)
    init += 100
  if GPU3 == True:
    t3 = thread(3,data,train,val,init,date)
  threads = []
  if GPU0 == True:
    print 'calling gpu thread 0'
    t0.start()
    time.sleep(1)
    threads.append(t0)
  if GPU1 == True:
    print 'calling gpu thread 1'
    t1.start()
    time.sleep(1)
    threads.append(t1)
  if GPU2 == True:
    print 'calling gpu thread 2'
    t2.start()
    time.sleep(1)
    threads.append(t2)
  if GPU3 == True:
    print 'calling gpu thread 3'
    t3.start()
    threads.append(t3)
  for t in threads:
    t.join()
  



