#-*- coding:utf-8 -*-
import sys, gzip, json, os, math, pickle, re
import numpy as np
from datetime import datetime
from math import exp, log, sqrt


root_path = '/home/marsan/workspace/tnative'
sys.path.append(root_path)
from lib import dgen as dgen
from lib import grader as grader
from lib import learner as learner

# stable_model_path = root_path+"/models/stable.json.gz"

#==========================================
#   machein learning flow wrapper
#==========================================  
class ml(object):
  def __init__(self, alg, params={}, D=2**24, en_pnorm=False, interaction=False, 
                feature_select=[], feature_drop=[], remove_blacklist=False, layer2_ensemble=None,
                en_fast_data=False, en_fast_load=False, en_plot=False, save=False, save_fmt='pkl', 
                dump_y2p_ids=False, debug=False
              ):

    # feature parameters
    self.alg = alg
    self.params = params
    self.D = D
    self.interaction = interaction
    
    # feature selection
    self.feature_select = feature_select
    self.feature_drop = feature_drop
    self.remove_blacklist = remove_blacklist
    self.layer2_ensemble = layer2_ensemble

    # ctrl
    self.en_plot = en_plot
    self.en_fast_data = en_fast_data
    self.en_fast_load = en_fast_load
    self.en_pnorm = en_pnorm
    self.save = save
    self.save_fmt = save_fmt
    self.dump_y2p_ids = dump_y2p_ids
    self.filepath = '_'
    self.debug = debug

    # sub-modules
    self.dgen = dgen.data_gen(D=self.D, en_fast_data=self.en_fast_data, en_fast_load=self.en_fast_load, interaction=self.interaction, feature_select=self.feature_select, feature_drop=self.feature_drop, remove_blacklist=self.remove_blacklist, layer2_ensemble=self.layer2_ensemble, debug=self.debug)
    self.grader = grader.grader(en_plot=en_plot)
    if self.alg == 'xgboost': 
      params['early_stop'] = True
      params['e_round'] = 10
      params['eval_set'] = self.dgen.gen_data(0.85, 0.9)
      self.dgen.tbl = self.dgen.tbl.filter(rand__lt=0.85)
      print('[NOTE] for xgboost early_stop, protect dgen sample rand=0.85-0.90 as watchlist.')

    self.learner = learner.learner(alg=self.alg, params=self.params, D=self.D)
    
    print("\n[%s] D=2**%i, params=%s, en_pnorm=%s, interaction=%i, remove_blacklist=%s, en_fast_data=%s, en_plot=%s, save=%i, save_fmt=%s" % \
      (self.alg, math.log(D, 2), self.learner.params, self.en_pnorm, self.interaction, self.remove_blacklist, self.en_fast_data, self.en_plot, self.save, self.save_fmt))


  #-------------------------
  #   train & test
  #-------------------------
  def learn_samples(self, rmin, rmax, info='', pnorm=None, vcnt=1000):
    training = (pnorm == None)
    # model work
    raw = self.dgen.gen_data(rmin, rmax, hashing=self.alg)
    if self.alg not in ['ffm', 'sksgd']:
      true_cnt, false_cnt, all_cnt = None, None, None   # for saving time
    else:
      true_cnt, false_cnt, all_cnt = self.dgen.observe_data(rmin, rmax, silent=True)
    y2p = self.learner.train(raw, training, {'all_cnt': all_cnt})
    # performance
    if self.en_pnorm:
      pnorm = pnorm if pnorm else self.grader.find_pnorm(y2p)
      y2p = [[y, min(1, p/pnorm)] for y, p in y2p]
    else:
      pnorm = 1
    if self.dump_y2p_ids: # dump y2p & ids
      ids = [str(r.id) for r in self.dgen.raw_range(rmin, rmax).only('id')]
      y2p_fname = "%s_%s_%s.pkl" % (self.dump_y2p_ids, training, datetime.now().strftime("%Y%m%d_%H%M"))
      pickle.dump([y2p, ids], open(y2p_fname, 'wb'))
      print("dump_y2p_ids in %s" % y2p_fname)
    auc = self.grader.auc_curve(y2p)
    scan = self.grader.scan_all_threshold(y2p)
    return auc, pnorm, scan, {'true_cnt': true_cnt, 'false_cnt': false_cnt, 'all_cnt': all_cnt}


  def train(self, tmin, tmax, vmin=None, vmax=None, skip_test=False):
    Ts = datetime.now()
    if (vmin == None):
      vmin = (1-tmax) if (tmax < 0.5) else tmax
      vmax = 1
    model_id = None
    print("\n%s\n#  [Train %s] %.2f - %.2f @ %s\n%s" % ("-"*60, self.alg, 100*tmin, 100*tmax, datetime.now(), "-"*60))
    _, pnorm, _, train_info = self.learn_samples(tmin, tmax, info='train')
    if skip_test: return  # while ensemble, no need to test each model
    try:
      print("\n%s\n#  [Test %s] %.2f - %.2f @ %s\n%s" % ("-"*60, self.alg, 100*vmin, 100*vmax, datetime.now(), "-"*60))
      auc, _, scan, test_info = self.learn_samples(vmin, vmax, info='test', pnorm=pnorm)
      print(('finished, pnorm=%f, test_auc = %.8f, elapsed time: %s' % (pnorm, auc, str(datetime.now() - Ts))))
      if self.save:
        self.filepath = "%s/models/%s_t%i_v%i_auc_%i_%s" % (root_path, self.alg, (tmax-tmin)*100, (vmax-vmin)*100, auc*1000, datetime.now().strftime("%Y%m%d_%H%M"))
        features = self.en_fast_data if self.en_fast_data else list(self.dgen.rand_sample(hashing=False)[1].keys())
        info = {
          'D': self.D,
          'dim': self.learner.dim,
          'features': features,
          'srate': (tmax-tmin),
          'train_sample_cnt': train_info['all_cnt'],
          'train_clk_cnt': train_info['true_cnt'], 
          'vrate': (vmax-vmin),
          'test_sample_cnt': test_info['all_cnt'],
          'test_clk_cnt': test_info['true_cnt'],
          'perfactor': 'auc',
          'performance': auc,
          'scan': scan,
          'pnorm': pnorm,
          'ext_fname': self.filepath,
        }
        if self.save_fmt == 'release':
          self.learner.release_model(info, filepath=self.filepath)
          self.verify_model(self.filepath) # try to reproduce
        else:
          self.filepath = self.learner.save_model(self.save_fmt, info, filepath=self.filepath)
        print("[model saved] in %s" % (self.filepath))
    except Exception as e:
      filepath = "%s/models/%s_t%i_v%i_tmp_%s" % (root_path, self.alg, (tmax-tmin)*100, (vmax-vmin)*100, datetime.now().strftime("%Y%m%d_%H%M"))
      self.learner.release_model({}, filepath=filepath)
      print("[ERROR] in test phase, save training model in %s\n %s" % (filepath, e))
    return auc


  def verify_model(self, filepath):
    filepath = filepath.split('.')[0]
    print("[verify model decision_function]:")
    # [pickled correct model]
    pkl_model, pinfo = pickle.load(open(filepath+'.pkl', 'rb'))
    # [loaded model]
    data = json.loads(gzip.open(filepath+'.json.gz', 'rt').read())
    os = data['intercept']
    ws = json.loads(data['weights'])
    print('os', os)
    for idx in range(10):
      # take sample
      t1y, t1x = self.dgen.rand_sample(hashing=True)
      if self.alg in ['sklr', 'sksgd']:
        Xt = [0]*self.D
        for i in t1x: Xt[i] = 1
        yp_real = pkl_model.decision_function(Xt)[0]
        yp_load = sum([ws.get(str(x), 0) for x in t1x]) + os
      elif self.alg in ['ftrl']:
        yp_real = pkl_model.predict(t1x, decision_func=True)
        yp_load = sum([ws.get(str(x), 0) for x in t1x]) + ws['0']
      else:
        print("ohoh, verify_model does not support alg=%s yet ..." % self.alg)
      print("[%i] correct: %f%% / reproduced: %f%%, diff=%s" % (idx, yp_real, yp_load, (yp_real - yp_load)))



#==========================================
#   test main
#==========================================
if __name__ == '__main__':
  ml(alg='sklr', en_plot=False, save=True).train(tmin=0, tmax=0.1)  

