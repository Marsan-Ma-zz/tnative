import sys, gzip, json, os, math, pickle, re, subprocess
import scipy.sparse as sparse
import numpy as np
from datetime import datetime
from math import exp, log, sqrt
import multiprocessing as mp
pool_size = int(mp.cpu_count())

root_path = '/home/marsan/workspace/tnative'
sys.path.append(root_path)

# models
from lib import schema as db
from lib import dgen as dgen
from lib import grader as grader
  
#==========================================
#   main class
#==========================================  
class libffm(object):
  def __init__(self, lambd, k, itera, eta, fold, nr_threads=pool_size):
    # params
    self.lambd        = lambd         # regularization cost
    self.k            = k             # latent factors
    self.iter         = itera         # iterations
    self.eta          = eta           # learning rate
    self.nr_threads   = nr_threads    # threads
    self.fold         = fold          # n folds
    # io files
    timestamp         = datetime.now().strftime("%Y%m%d_%H%M")
    self.workpath     = root_path+'/'
    self.feat_set     = dgen.data_gen(submit=False).info_set()
    self.io_train     = self.workpath+'tmp/bigdata.tr.%s.dat' % self.feat_set
    self.io_valid     = self.workpath+'tmp/bigdata.va.%s.dat' % self.feat_set
    self.io_test      = self.workpath+'tmp/bigdata.te.%s.dat' % self.feat_set
    self.io_model     = self.workpath+'models/ffm_%s.model' % timestamp
    self.io_out_tr    = self.workpath+'tmp/output_%s.tr.dat' % timestamp
    self.io_out_te    = self.workpath+'tmp/output_%s.te.dat' % timestamp
    # bin files
    self.exec_train   = self.workpath+'lib/libffm/ffm-train'
    self.exec_predict = self.workpath+'lib/libffm/ffm-predict'

  #-------------------------
  #   tasks
  #-------------------------
  def write_samples(self, raw, filename, fvalid=None, valid_idx=-1):
    if os.path.isfile(filename):  
      # if file exists with same hash number, content should be the same, skip dump
      print("file %s exists, skip dump." % filename)
    else:
      if fvalid: fva = open(fvalid, 'wt')
      with open(filename, 'wt') as f:
        for data in raw:
          idx, x, y = data[0], data[1], data[2]
          # print(y, x)
          feats = ["%i:%i:%.10f" % (field, feat, val) for field, feat, val in x]
          line = ("%i " % (-1 if y == None else y)) + " ".join(feats) + "\n"
          if ((fvalid != None) & (valid_idx > 0) & (idx >= valid_idx)):
            fva.write(line)
          else:
            f.write(line)
          if (idx % 10000 == 0): print("ffm write file: %i @ %s" % (idx, datetime.now()))
      if fvalid: fva.close()
          

  def read_results(self, filename):
    with open(filename, 'rt') as f:
      lines = f.readlines()
    return [float(p.split(' ')[0]) for p in lines]


  def predict_proba(self, test_file, out_file):
    cmd = [self.exec_predict, test_file, self.io_model, out_file]
    print("[ffm predict_proba cmd]", " ".join(cmd))
    stdout = subprocess.check_output(cmd)
    print("predict_proba stdout:", str(stdout))
    y_list = self.read_results(test_file)
    p_list = self.read_results(out_file)
    y2p = zip(y_list, p_list)
    return list(y2p)


  #-------------------------
  #   main flow
  #-------------------------
  def fit(self, raw, all_cnt, early_stop=None):
    self.io_train = self.io_train+'.'+str(all_cnt)
    self.io_valid = self.io_valid+'.'+str(all_cnt)
    cmd = [self.exec_train, '-l', "%.8f" % self.lambd, '-k', str(self.k), '-s', str(self.nr_threads), '-r', str(self.eta), '-t', str(self.iter)]
    if (self.fold > 1): cmd += ['-v', str(self.fold)]
    if early_stop:
      cmd += ['-p', self.io_valid, '--auto-stop']
      self.write_samples(raw, self.io_train, fvalid=self.io_valid, valid_idx=int(all_cnt*(1-early_stop)))
    else:
      # cmd += ['-t', str(self.iter)]
      self.write_samples(raw, self.io_train)
    cmd += [self.io_train, self.io_model]
    print("[ffm fit cmd]", " ".join(cmd))
    stdout = subprocess.check_output(cmd)
    print("[fit stdout]", '-'*20, "\n")
    for l in [ line.split() for line in str(stdout).split("\\n")]: print(' '.join(l))
    print("-"*32)
    train_y2p = self.predict_proba(test_file=self.io_train, out_file=self.io_out_tr)
    return list(train_y2p)

  def test(self, raw, all_cnt=None):
    self.io_test = self.io_test+'.'+str(all_cnt)
    self.write_samples(raw, self.io_test)
    test_y2p = self.predict_proba(test_file=self.io_test, out_file=self.io_out_te)
    return list(test_y2p)


  #-------------------------
  #   self-testing
  #-------------------------
  def test_ffm(self, en_plot=False):
    self.io_train  = self.workpath+'lib/libffm/bigdata.tr.txt'
    self.io_test   = self.workpath+'lib/libffm/bigdata.te.txt'
    grd = grader.grader(en_plot=en_plot)
    #
    y2p_train = self.fit(raw=None)
    auc_train = grd.auc_curve(y2p_train)
    y2p_test = self.test(raw=None)
    auc_test = grd.auc_curve(y2p_test)
    scan = grd.scan_all_threshold(y2p_test)
    return y2p_train, y2p_test


#==========================================
#   verify
#==========================================
if __name__ == '__main__':
  libffm().test_ffm(en_plot=False)
