import sys, gzip, json, os, math, pickle, re, copy
import numpy as np
import multiprocessing as mp
pool_size = int(mp.cpu_count())

from datetime import datetime
from math import exp, log, sqrt


root_path = '/home/marsan/workspace/tnative'
sys.path.append(root_path)
from lib import dgen as dgen
from lib import top as top

#==========================================
#   machein learning flow wrapper
#==========================================  
# [container of model unit]
class ml_unit(object):
  def __init__(self, alg, D, data, tmin, tmax, en_plot, en_fast_data, skip_test, debug):
    self.ml = top.ml(alg=alg, D=D, en_plot=en_plot, en_fast_data=en_fast_data, debug=debug)
    self.data = data
    self.tmin = tmin
    self.tmax = tmax
    self.vrng = abs(tmax - tmin)
    self.vmin = 1 - self.vrng if self.vrng < 0.5 else self.vrng
    self.vmax = 1
    self.y2p = []
    self.pnorm = -1
    self.skip_test = skip_test
    # self.q = mp.Queue()

    # update data filter
    for fea, val in self.data.items():
      self.ml.dgen.tbl = eval("self.ml.dgen.tbl.filter(%s=val)" % (fea))
    
  # for multiprocess
  def train_unit(self):
    self.ml.train(self.tmin, self.tmax, self.vmin, self.vmax, skip_test=self.skip_test)
    # q.put([self.ml.learner])
    return self.ml.learner


class ensembler(object):
  def __init__(self, segments, vrate, D=2**24, en_fast_data=None, en_plot=False, en_pnorm=False, skip_test=False, debug=False):
    # samples
    self.D = D
    self.vmin = 1 - vrate
    self.vmax = 1
    
    # ctrl
    self.en_plot = en_plot
    self.en_pnorm = en_pnorm
    self.skip_test = skip_test
    self.en_fast_data = en_fast_data
    self.debug = debug
    
    # initialize models 
    self.dgen = dgen.data_gen(D=self.D, en_fast_data=self.en_fast_data, debug=self.debug) # samples for final test
    self.ml_group = []
    for s in segments:
      item = ml_unit(alg=s['alg'], D=self.D, data=s['data'], tmin=s['tmin'], tmax=s['tmax'], en_plot=self.en_plot, en_fast_data=self.en_fast_data, skip_test=self.skip_test, debug=self.debug)
      self.ml_group.append(item)
      

  #-------------------------
  #   convension
  #-------------------------
  def finitize(self, n, e=35):
    return max(min(n, e), -e)   # make -e <= n <= e

  def merge_sigmoid(self, nlist, e=35):
    nlist = [-log(max((1/max(n, 1e-35) - 1), 1e-35)) for n in nlist]
    nmean = np.mean(nlist)
    nsig = 1. / (1. + exp(-self.finitize(nmean, e)))
    return nsig


  #-------------------------
  #   train & test
  #-------------------------
  def train_and_test(self, en_multi_threads=False):
    self.train_all(en_multi_threads)
    roc_auc, yr_ens, yp_ens = self.test()
    self.save(roc_auc)
    return roc_auc, yr_ens, yp_ens


  def train_all(self, en_multi_threads=True):
    processes = []
    mp_pool = mp.Pool(pool_size)
    for l in self.ml_group:
      if not en_multi_threads:
        l.ml.train(l.tmin, l.tmax, l.vmin, l.vmax, skip_test=self.skip_test)  # [single process for debug]
      else:
        p = mp_pool.apply_async(l.train_unit, ())
        processes.append((l, p))
        # p = mp.Process(target=l.train_unit, args=(l.q,))
        # processes.append(p)
        # p.start()
    if en_multi_threads:
      for l, p in processes:
        l.ml.learner = p.get()
      # for l in self.ml_group:
      #   l.ml.learner = l.q.get()[0]
      # for p in processes: p.join()
    print("[Ensembler] models training done @ %s" % datetime.now())
    return self.ml_group


  def test(self):
    print("\n%s\n#  [Ensembler] start grader %.2f - %.2f @ %s\n%s" % ("-"*60, 100*self.vmin, 100*self.vmax, datetime.now(), "-"*60))
    yr_ens = {}
    yp_ens = {}
    for s in self.ml_group:
      sdgen = copy.copy(self.dgen)
      for fea, val in s.data.items():
        sdgen.tbl = eval("sdgen.tbl.filter(%s=val)" % (fea))
      ids = [str(r.id) for r in sdgen.raw_range(self.vmin, self.vmax).only('id')]
      raw = sdgen.gen_data(self.vmin, self.vmax)
      # get y2p
      s.y2p = s.ml.learner.train(raw, training=False, info={'all_cnt': -1})
      if self.en_pnorm:
        s.pnorm = s.ml.grader.find_pnorm(s.y2p)
        s.y2p = [[y, min(1, p/s.pnorm)] for y, p in s.y2p]
      # map to ensembled y2p
      for i in range(len(ids)):
        key = ids[i]
        if key not in yp_ens: 
          yr_ens[key] = []
          yp_ens[key] = []
        yr_ens[key].append(s.y2p[i][0])
        yp_ens[key].append(s.y2p[i][1])

    y2p_ens = [(np.mean(yrs), self.merge_sigmoid(yp_ens[rid])) for rid, yrs in yr_ens.items()]
    grader = self.ml_group[0].ml.grader
    roc_auc = grader.auc_curve(y2p_ens)
    scan = grader.scan_all_threshold(y2p_ens)
    print("[Ensembler] ensembled ROC: %.3f%% @ %s" % (roc_auc*100, datetime.now()))
    return roc_auc, yr_ens, yp_ens

  #-------------------------
  #   layer-2
  #-------------------------
  def train_layer2(self):
    # collect samples
    Xt = []
    Yt = []
    sdgen = copy.copy(self.dgen)
    pass


  def test_layer2(self):
    pass

  #-------------------------
  #   model reuse
  #-------------------------
  def save(self, auc):
    filepath = "%s/models/m%i_v%i_auc_%i_%s" % (root_path, len(self.ml_group), (self.vmax-self.vmin)*100, auc*1000, datetime.now().strftime("%Y%m%d_%H%M"))
    trained_models = [mlu.ml.learner for mlu in self.ml_group]
    pickle.dump(trained_models, open(filepath, 'wb'))
    print("ensemble model saved in %s @ %s" % (filepath, datetime.now()))
    return filepath


#==========================================
#   experiments
#==========================================
def k_fold_ensemble(alg, k, vrate=0.1, en_plot=False, en_fast_data=None, skip_compare=True):
  if not skip_compare:
    print("="*5, '[train by single thread]', '='*40)
    top.ml(alg=alg, en_plot=en_plot).train(0, 1-vrate)
  print("[%i_fold_%s_ensemble] start @ %s" % (k, alg, datetime.now()))
  segments = []
  step = (1.0 - vrate)/k
  for i in range(k):
    segments.append({
      'alg': alg,
      'tmin': step*(i+1),
      'tmax': step*i,
      'data': {
        'isad__ne': None,
        # 'label__ne': None,
      }
    })
  ens = ensembler(segments, vrate=vrate, en_plot=en_plot, en_fast_data=en_fast_data, skip_test=True)  # k-fold MUST skip_test=True!! since we block ens.vmin
  for item in ens.ml_group: # prevent models from getting test samples
    item.ml.dgen.tbl = item.ml.dgen.tbl.filter(rand__lt=ens.vmin) 
  return ens.train_and_test()

def xgboost_sklr(vrate=0.1, en_plot=False):
  segments = [
    {
      'alg': alg, 
      'tmin': 0, 
      'tmax': 1-vrate,
      'data': {
        'isad__ne': None,
        # 'label__ne': None,
      },
    } for alg in ['sklr', 'xgboost']
  ]
  return ensembler(segments, vrate=vrate, en_plot=en_plot).train_and_test()


#==========================================
#   dnq ensemble (divide-and-conquer)
#==========================================
dnq_segments = {
  'status'  : [('status', 'bad'), ('status', 'normal')],
  'lang'    : [('meta_lang__icontains', 'en'), ('meta_lang', '')],
  'domain'  : [('domain__icontains', '.net'), ('domain__icontains', '.com'), ('domain__icontains', '.org'), ('domain__icontains', '.uk')],
}
def dnq_ensemble(alg, segname, vrate=0.1, en_plot=False, skip_compare=True):
  srate = 1-vrate if vrate > 0.5 else vrate
  if not skip_compare:
    print("="*5, '[train by single thread]', '='*40)
    auc_single = top.ml(alg='sklr', en_plot=en_plot).train(0, srate)
  print("="*5, '[train by ensemble]', '='*40)
  if segname == 'all':
    segs = ([('fid__ne', -1)] + [j for k in dnq_segments.values() for j in k])
  else:
    segs = ([('fid__ne', -1)] + dnq_segments[segname])
  segments = [{
    'alg': alg, 
    'tmin': 0, 
    'tmax': srate,
    'data': {
      'isad__ne': None,
      s: v,
    },
  } for s,v in segs]
  print("[dnq_ensemble] condition: ", segments)
  ens = ensembler(segments, vrate=vrate, en_plot=en_plot)
  return ens.train_and_test()


#==========================================
#   verify
#==========================================
if __name__ == '__main__':
  cmd = str(sys.argv[1])
  vrate = float(sys.argv[2])
  if (len(sys.argv) >= 4): cmd2 = str(sys.argv[3])
  #
  if cmd == '5_fold_xgboost':
    k_fold_ensemble('xgboost', k=5, vrate=vrate, en_plot=False)
  elif cmd == '5_fold_sklr':
    k_fold_ensemble('sklr', k=5, vrate=vrate, en_plot=False, en_fast_data='D_20_tfidf_cnts')
  elif cmd == 'xgboost_sklr':
    xgboost_sklr(vrate=vrate, en_plot=False)
  elif cmd == 'dnq_ensemble':
    dnq_ensemble('sklr', cmd2, vrate=vrate, en_plot=False)    
  elif cmd == 'dnq_all':
    dnq_ensemble('sklr', 'all', vrate=vrate, en_plot=False)    



