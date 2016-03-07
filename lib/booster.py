import sys, gzip, json, os, math, pickle, re, copy
import numpy as np
import multiprocessing as mp
pool_size = int(mp.cpu_count())

from datetime import datetime
from math import exp, log, sqrt


root_path = '/home/marsan/workspace/tnative'
sys.path.append(root_path)
from lib import schema as db
from lib import dgen as dgen
from lib import top as top

#==========================================
#   machein learning flow wrapper
#==========================================  
# [container of model unit]
class booster(object):
  def __init__(self, D=2**24, en_fast_data=False):
    self.D = D
    self.en_fast_data = en_fast_data
    
  # model as 1st layer
  def embed_model_results(self, params, debug=False, clear=False):
    mp_pool = mp.Pool(pool_size)
    rmax = 0.01 if debug else 1
    field_tbl = {m['model_name']: m['field'] for m in params}
    # prepare samples
    print("[embed_model_results] start @ %s" % (datetime.now()))
    dg_submit = dgen.data_gen(D=self.D, en_fast_data=False, en_fast_load=True, submit=True)
    dg_train = dgen.data_gen(D=self.D, en_fast_data=self.en_fast_data, en_fast_load=False, submit=False)
    samples = list(dg_train.gen_data(0, rmax, hashing=True, extra=True)) + list(dg_submit.gen_data(0, rmax, hashing=True, extra=True))
    if 'ffm' in " ".join(field_tbl.keys()):
      print('include ffm model, need special sample format.')
      samples_ffm = list(dg_train.gen_data(0, rmax, hashing=False, extra=True)) + list(dg_submit.gen_data(0, rmax, hashing=False, extra=True))
    fids = [extra['fid'] for idx, x, y, extra in samples]
    all_cnt = len(samples)
    print("[embed_model_results] prepare %i samples done @ %s" % (all_cnt, datetime.now()))
    # predicting by all models
    results = {}
    processes = []
    for para in params:
      learner, info = pickle.load(open('%s/models/%s' % (root_path, para['model_pkl']), 'rb'))
      samples_use = samples_ffm if ('ffm' in para['model_name']) else samples
      samples_gen = ((idx, x, y) for idx, (odx, x, y, fid) in enumerate(samples_use))
      print("---------- start predict by %s @ %s ----------" % (para['model_name'], datetime.now()))
      y2p = learner.train(samples_gen, training=False, info={'all_cnt': all_cnt})
      print("---------- finish predict by %s @ %s ----------" % (para['model_name'], datetime.now()))
      # proc = mp_pool.apply_async(learner.train, (samples_gen, False, {'all_cnt': all_cnt}))
      # processes.append([para, proc])
    # for para, proc in processes:
      # y2p = proc.get()
      results[para['model_name']] = {fid: pp for (fid, (yy, pp)) in zip(fids, y2p)}
      print("done predicting %s @ %s" % (para['model_name'], datetime.now()))
    # save results
    pickle.dump(results, open(root_path+'/data/embed_model_sklr.predicts.pkl', 'wb'))
    # update samples
    # processes = []
    # for idx, fid in enumerate(fids):
    #   d = db.articles.objects(fid=fid).first()
    #   tbl = {} if clear else getattr(d, field_tbl[key])
    #   for key, table in results.items():
    #     tbl[key] = table[fid]
    #   # exec("d.update(%s=tbl)" % field_tbl[key])
    #   p = mp_pool.apply_async(thread_update, (d, field_tbl[key], tbl))
    #   processes.append(p)
    #   if (idx % 10000 == 0):
    #     for p in processes: r = p.get()
    #     print("update %i records: fid=%s @ %s" % (idx, fid, datetime.now()))
    # print("[embed_model_results] all done @ %s" % (datetime.now()))
    # return fids, samples, results


#------------------------------
#   multitask bottom blocks
#------------------------------
def thread_update(rec, key, val):
  exec("rec.update(%s=val)" % key)

#==========================================
#   verify
#==========================================
if __name__ == '__main__':
  # parse cmd
  cmd = {}
  for i in range(len(sys.argv)):
    cmd[i] = sys.argv[i]
  print(cmd)

  # embed model results
  if (cmd[1] == 'layer_1_sklr'):
    params = [
      # [sklr]
      {'model_pkl': 'sklr_t30_v9_auc_931_20151002_1653.pkl', 'model_name': 'sklr_0', 'field': 'ensemble_0'},
      {'model_pkl': 'sklr_t30_v9_auc_930_20151002_1657.pkl', 'model_name': 'sklr_1', 'field': 'ensemble_0'},
      {'model_pkl': 'sklr_t30_v9_auc_931_20151002_1701.pkl', 'model_name': 'sklr_2', 'field': 'ensemble_0'},
      {'model_pkl': 'sklr_t30_v9_auc_930_20151002_1705.pkl', 'model_name': 'sklr_3', 'field': 'ensemble_0'},
      {'model_pkl': 'sklr_t29_v9_auc_930_20151002_1708.pkl', 'model_name': 'sklr_4', 'field': 'ensemble_0'},
      {'model_pkl': 'sklr_t30_v9_auc_930_20151002_1712.pkl', 'model_name': 'sklr_5', 'field': 'ensemble_0'},
      {'model_pkl': 'sklr_t29_v9_auc_930_20151002_1716.pkl', 'model_name': 'sklr_6', 'field': 'ensemble_0'},
      # # [xgboost]
      # {'model_pkl': 'xgboost_t50_v9_auc_953_20151001_1134.pkl', 'model_name': 'xgboost_953_0', 'field': 'ensemble_1'},
      # {'model_pkl': 'xgboost_t50_v9_auc_953_20151001_1304.pkl', 'model_name': 'xgboost_953_1', 'field': 'ensemble_1'},
      # {'model_pkl': 'xgboost_t49_v9_auc_952_20151001_1436.pkl', 'model_name': 'xgboost_952_2', 'field': 'ensemble_1'},
      # {'model_pkl': 'xgboost_t50_v9_auc_952_20151001_1605.pkl', 'model_name': 'xgboost_952_3', 'field': 'ensemble_1'},
      # {'model_pkl': 'xgboost_t50_v9_auc_953_20151001_1735.pkl', 'model_name': 'xgboost_953_4', 'field': 'ensemble_1'},
      # [ffm]
      # {'model_pkl': 'ffm_t60_v9_auc_940_20150929_1900.pkl', 'model_name': 'ffm_936_0', 'field': 'ensemble_2'},
    ]
    booster(D=2**24, en_fast_data='D_20_tfidf_cnts').embed_model_results(params, clear=True)

