import sys, gzip, json, os, math, pickle, re, shutil
import scipy.sparse as sparse
import numpy as np
from datetime import datetime
from math import exp, log, sqrt
from sklearn.cross_validation import cross_val_predict
root_path = '/home/marsan/workspace/tnative'
sys.path.append(root_path)

# models
import xgboost as xgb
from sklearn import linear_model, ensemble, grid_search
from sklearn.cross_validation import StratifiedKFold
from sklearn.cross_validation import cross_val_predict
from sklearn.feature_selection import RFECV

from lib import schema as db
from lib import ftrl as ftrl
from lib import ffm as ffm

#==========================================
#   main class
#==========================================  
class learner(object):
  def __init__(self, alg, params={}, D=2**24, interaction=False):
    # feature parameters
    self.alg = alg
    self.params = params
    self.D = D
    self.interaction = interaction

    # params defaults
    if self.alg == 'ffm':
      self.params = {'lambd': 1e-5, 'k': 2, 'itera': 150, 'eta': 0.05, 'fold': 1}
    if self.alg == 'xgboost':
      self.params = {'max_depth': 15, 'learning_rate': 0.15, 'n_estimators': 1000, 'silent': False, 'eval_metric': 'auc'}
      # self.params = {'max_depth': 7, 'learning_rate': 0.01, 'n_estimators': 25000, 'silent': False, 'eval_metric': 'auc'}
    elif self.alg == 'ftrl':
      self.params = {'alpha': 0.01, 'beta': 0.1, 'L1': 1.0, 'L2': 1.0, 'epoch': 1}
    elif self.alg == 'skgbdt':
      self.params = {'learning_rate': 0.1, 'n_estimators': 100}
    elif self.alg == 'skrf':
      self.params = {'n_estimators': 300, 'n_jobs': -1}
    elif self.alg == 'sklr':
      self.params = {'C': 0.2, 'penalty': 'l2', 'dual': False, 'tol': 1e-6, 'max_iter': 100, 'rfecv': False, 'cv': None}
      # self.params = {'C': 0.2, 'penalty': 'l2', 'dual': False, 'tol': 1e-8, 'max_iter': 100, 'rfecv': False, 'cv': 4}
    elif self.alg == 'sksgd':
      self.params = {'alpha': 0.0001, 'l1_ratio': 0.15, 'epoch': 5, 'epsilon': 0.1}

    for k, v in params.items(): self.params[k] = v

    # info cache
    self.dim = -1   # record dimention

    # main modules
    sp = self.params
    if self.alg == 'ffm':
      self.learner = ffm.libffm(lambd=sp['lambd'], k=sp['k'], itera=sp['itera'], eta=sp['eta'], fold=sp['fold'])
    elif self.alg == 'xgboost':
      self.learner = xgb.XGBClassifier(max_depth=sp['max_depth'], learning_rate=sp['learning_rate'], n_estimators=sp['n_estimators'], silent=sp['silent'])
    elif self.alg == 'ftrl':
      self.learner = ftrl.ftrl_proximal(sp['alpha'], sp['beta'], sp['L1'], sp['L2'], self.D, self.interaction)
    elif self.alg == 'skgbdt':
      self.learner = ensemble.GradientBoostingClassifier(learning_rate=sp['learning_rate'], n_estimators=sp['n_estimators'])
    elif self.alg == 'skrf':
      self.learner = ensemble.RandomForestClassifier(n_estimators=sp['n_estimators'], n_jobs=sp['n_jobs'])
    elif self.alg == 'sklr':
      self.learner = linear_model.LogisticRegression(C=sp['C'], penalty=sp['penalty'], dual=sp['dual'], tol=sp['tol'], max_iter=sp['max_iter'])
    elif self.alg == 'sksgd':
      self.learner = linear_model.SGDClassifier(alpha=sp['alpha'], l1_ratio=sp['l1_ratio'], n_iter=sp['epoch'], n_jobs=7)


  #-------------------------
  #   conventions
  #-------------------------
  def sigmoid(self, wx):
    return 1. / (1. + exp(-max(min(wx, 35.), -35.)))

  def raw2sparseXY(self, raw):
    Ts = datetime.now()
    # Xt, Yt = [(0, self.D-1, 0)], [] # dummy feature for telling model max-dimention
    Xt, Yt = [(0, self.D-1+10000, 0)], [] # dummy feature for telling model max-dimention
    # for idx, x, y in raw:
    for data in raw:
      idx, x, y = data[0], data[1], data[2]
      Yt.append(y)
      Xt += [(idx, col, val) for col, val in x]
      if (idx % 10000 == 0): print(idx, datetime.now())
    print('shape(Xt): ', np.shape(Xt))
    row, col, val = list(zip(*Xt))
    Xt = sparse.coo_matrix((val, (row, col))).tocsr()
    print("raw2xy cost %s secs to prepare %i samples." % ((datetime.now() - Ts), len(Yt)))
    return Xt, Yt

  #-------------------------
  #   models work
  #-------------------------
  def train(self, raw, training, info={}):
    if self.alg == 'ffm':
      y2p = self.train_ffm(raw, training, all_cnt=info.get('all_cnt'))
    elif self.alg == 'xgboost':
      y2p = self.train_xgboost(raw, training)
    elif self.alg == 'ftrl':
      y2p = self.train_ftrl(raw, training)
    elif self.alg == 'skgbdt':
      y2p = self.train_skgbdt(raw, training)
    elif self.alg == 'skrf':
      y2p = self.train_skrf(raw, training)
    elif self.alg == 'sklr':
      y2p = self.train_sklr(raw, training)
    elif self.alg == 'sksgd':
      y2p = self.train_sksgd(raw, training, info['all_cnt'])
    return y2p


  def train_sklr(self, raw, training):
    Xt, Yt = self.raw2sparseXY(raw)
    if training: 
      if self.params['rfecv']:
        rfecv = RFECV(estimator=self.learner, step=0.1, cv=StratifiedKFold(Yt, 4), scoring='roc_auc', verbose=True)
        rfecv.fit(Xt, Yt)
        print("[RFECV]: optimal num_of_feats:%i, support_=%s @ %s" % (rfecv.n_features_, rfecv.support_, datetime.now()))
        self.learner = rfecv
      elif self.params['cv']:
        gs_params = {'C': [m*self.params['C'] for m in [0.5, 1, 2]]}
        print("[grid_search] for sklr with gs_params=%s" % gs_params)
        gs = grid_search.GridSearchCV(self.learner, gs_params, cv=4)
        gs.fit(Xt, Yt)
        self.learner = gs
      else:
        self.learner.fit(Xt, Yt)
    y2p = [[y, self.sigmoid(self.learner.decision_function(Xt[idx])[0])] for idx, y in enumerate(Yt)]
    # if not self.params['rfecv']:
    #   self.dim = len([w for w in self.learner.coef_[0].tolist()])
    return y2p

  def train_ffm(self, raw, training, all_cnt=None):
    y2p = self.learner.fit(raw, all_cnt=all_cnt, early_stop=0.1) if training else self.learner.test(raw, all_cnt=all_cnt)
    return y2p

  def train_ftrl(self, raw, training, vcnt=10000):
    Ts = datetime.now()
    loss = 0.
    count = 0
    y2p = []
    for ep in range(self.params['epoch']):
      for idx, x, y in raw:
        p = self.learner.predict(x)
        y2p.append([y, p])
        if ((idx > 0) & (idx % vcnt == 0)):  #observe
          loss += ftrl.logloss(p, y)
          count += 1
          print(('[%i/%i] logloss: %f/%i=%f, elapsed time: %s' % (ep, idx, loss, count, loss/count, str(datetime.now() - Ts))))
        elif training:   # update only when training
          self.learner.update(x, p, y)
    # record info
    self.dim = len([w for w in self.learner.z if w > self.params.get('L1', 0.1)])
    return y2p

  def train_xgboost(self, raw, training):
    Xt, Yt = self.raw2sparseXY(raw)
    if training:
      if self.params.get('early_stop'):
        rnd = int(self.params['e_round'])
        Xe, Ye = self.raw2sparseXY(self.params['eval_set'])
        self.learner.fit(Xt, Yt, eval_metric=self.params['eval_metric'], early_stopping_rounds=rnd, eval_set=[(Xe, Ye)])
      else:
        self.learner.fit(Xt, Yt, eval_metric=self.params['eval_metric'])
    y2p = [[y, self.learner.predict_proba(Xt[idx])[0][1]] for idx, y in enumerate(Yt)]
    return y2p

  def train_skrf(self, raw, training):
    Xt, Yt = self.raw2sparseXY(raw)
    if training: self.learner.fit(Xt, Yt)
    y2p = [[y, self.learner.predict_proba(Xt[idx])[0][1]] for idx, y in enumerate(Yt)]
    return y2p

  def train_sksgd(self, raw, training, all_cnt, vcnt=1e4):
    Ts = datetime.now()
    Xt, Yt = [(0, self.D-1)], []
    y2p = []
    idx = 0  # for coordinate ctrl, cannot use loop index ! 
    for _, x, y in raw:
      Yt.append(y)
      Xt += [(idx, xx) for xx in x]
      if (((idx > 0) & (idx % vcnt == 0)) | (idx >= all_cnt-1)):
        row, col = list(zip(*Xt))
        Xt = sparse.coo_matrix(([1.]*len(row), (row, col))).tocsr()
        if training: self.learner.partial_fit(Xt, Yt, classes=[0.0, 1.0])
        print("sksgd cost %s secs to partial_fit %i samples." % ((datetime.now() - Ts), len(Yt)))
        y2p += [[y, self.sigmoid(self.learner.decision_function(Xt[idx2])[0])] for idx2, y in enumerate(Yt)]      
        Xt, Yt, idx = [(0, self.D-1)], [], 0
      else:
        idx += 1
    self.dim = len([w for w in self.learner.coef_[0].tolist()])
    return y2p

  #-------------------------
  #   save model
  #-------------------------
  def checkout_ftrl_weights(self):
    learner = self.learner
    L1 = learner.L1
    L2 = learner.L2
    alpha = learner.alpha
    beta = learner.beta
    weights = {}
    for idx, (z, n) in enumerate(zip(learner.z, learner.n)):
      sign = -1. if z < 0 else 1.  # get sign of z[i]
      if (sign * z > L1):
        weights[idx] = (sign * L1 - z) / ((beta + sqrt(n)) / alpha + L2)
    return weights
  

  def save_model(self, fmt, info={}, filepath=None):
    shutil.copyfile(root_path+'/lib/dgen.py', filepath+'.dgen')
    if fmt in ['pkl', 'pickle']:
      filepath += '.pkl'
      self.params['eval_set'] = None  # else will make self can't be pickled
      pickle.dump([self, info], open(filepath, 'wb'))
    elif fmt == 'json':
      filepath += '.json.gz'
      with gzip.open(filepath, 'wt') as f:
        if self.alg == 'ftrl':
          weights = self.checkout_ftrl_weights()
          intercept = 0
        elif ((self.alg == 'sklr') or (self.alg == 'sksgd')):
          weights = {idx: v for idx, v in enumerate(self.learner.coef_[0]) if v != 0}
          intercept = self.learner.intercept_[0]
        data = {
          'model'           : self.alg,
          'features'        : json.dumps(info.get('features')),
          'weights'         : json.dumps(weights),
          'intercept'       : intercept,
          'v2p_rng'         : [0, info.get('pnorm')],
          'scaler_mean'     : None,
          'scaler_std'      : None,
          'dday'            : info.get('vstart'),
          'ctime'           : str(datetime.now()),
        }
        f.write(json.dumps(data))
    else:
      print("[ERROR] unknown save format: %s" % fmt)
    return filepath

    
  def release_model(self, info={}, filepath=None):
    pkl_file = self.save_model('pickle', info, filepath)
    json_file = self.save_model('json', info, filepath)
    item = db.smodel(
      model_name      = self.alg,
      features        = json.dumps(info.get('features')),
      weights         = 'External',
      v2p_rng         = [0, info.get('pnorm')],
      scaler_mean     = None,
      scaler_std      = None,
      #
      info            = json.dumps(info),
      dday            = info.get('vstart'),
      ctime           = datetime.now(),
      stable          = True,
    )
    item.save()
    print("model saved, smodel_id=%s" % (item.id))
    return json_file

