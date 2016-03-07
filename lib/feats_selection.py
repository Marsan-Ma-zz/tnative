import sys, gzip, json, os, math, pickle, re, shutil, hashlib, operator
import scipy.sparse as sparse
import numpy as np
import matplotlib.pyplot as plt

from datetime import datetime
from math import exp, log, sqrt
from sklearn import linear_model, ensemble
from collections import Counter, OrderedDict

root_path = '/home/marsan/workspace/tnative'
sys.path.append(root_path)

from lib import dgen

#==========================================
#   main class
#==========================================  
class feats_selection(object):
  def __init__(self, params={}, D=2**24):
    self.D = D
    self.dg = dgen.data_gen(D=self.D)


  #---------------------------------
  #   convensions
  #---------------------------------
  


  #---------------------------------
  #   random feature as baseline
  #---------------------------------
  def probe_randoms(self, s_size=10000):
    fig = plt.figure(figsize=(18,15))
    ax1 = fig.add_subplot(3, 2, 1)
    ax2 = fig.add_subplot(3, 2, 2)
    ax3 = fig.add_subplot(3, 2, 3)
    # gen randoms
    s1 = np.random.lognormal(0, 1, s_size)
    s2 = np.random.normal(0, 1, s_size)
    s3 = np.random.uniform(-1, 1, s_size)
    # plot
    ax1.hist(s1, 100, normed=True, align='mid')
    ax2.hist(s2, 100, normed=True, align='mid')
    ax3.hist(s3, 100, normed=True, align='mid')

  #---------------------------------
  #   analysis models
  #---------------------------------
  def probe_sklr_weights(self, filename): #='/models/sklr_t95_v5_auc_947_20151007_1800.pkl'):
    # load model
    mdl = pickle.load(open("%s/models/%s" % (root_path, filename), 'rb'))
    feats = {k: v for k,v in enumerate(mdl[0].learner.coef_[0])}
    # 
    t1y, x_raw = self.dg.rand_sample(hashing=False)
    fidx = self.dg.fidx
    x = self.dg.hashing_obj(x_raw)
    feats_w = {}
    for k, hval, val in x:
      kk = "%s_%s" % (fidx[k], hval)
      kkh = self.dg.str2hvec(kk)
      feats_w[(k, hval)] = feats.get(kkh)
    sorted_feats_w = OrderedDict(sorted(feats_w.items(), key=lambda t: abs(t[1]), reverse=True))
    for k,v in sorted_feats_w.items():
      print("[%s] %.6f" % (k,v))
    return sorted_feats_w



