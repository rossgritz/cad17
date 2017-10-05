##
# For taking commands and running cases
##

import multiGPU as gpu

from optparse import OptionParser


def parser():
  parser = OptionParser()
  #SELECT GPU
  parser.add_option('--gpu',dest='gpu',default='0')
  #SELECT DATA TYPE
  parser.add_option('--data',dest='data',default='luna')
  #SELECT TRAIN DIR
  parser.add_option('--train',dest='train',default='/scr/data/nodules/luna/cand/init/')
  #SELECT VALIDATION DIR
  parser.add_option('--val',dest='val',default='/scr/data/nodules/luna/cand/val/')
  #IDENTIFY STARTING COUNT
  parser.add_option('--count',dest='init',default='0')
  #ADD DATE FOR OUTPUT PATH
  parser.add_option('--date',dest='date',default=None)
  return parser.parse_args()


def main():
  opts, args = parser()
  gpu.run(opts)


if __name__ == '__main__':
  main()
