import os, gc, glob, pickle, sys
import numpy as np
import pandas as pd

import sys
sys.path.append('/home/marsan/workspace/tnative')
from lib import grader as grader
from sklearn import linear_model, grid_search

os.chdir('/home/marsan/workspace/tnative/blending/stacking')

#==========================================
#   tasks
#==========================================
def get_sample_rand(path=None):
  if path:
    df_rinfo = pickle.load(open(path, 'rb'))
  else:
    rand_info = []
    pkl_folder = "/home/marsan/workspace/tnative/data/D_20_all"
    files = [ f for f in os.listdir(pkl_folder) if os.path.isfile(os.path.join(pkl_folder, f)) ]
    for f in files:
      samples = pickle.load(open("%s/%s" % (pkl_folder, f), 'rb'))
      for s in samples:
        ss = {
          'file': "%s_raw_html.txt" % s['fid'],
          'rand': s['rand'],
          'dans': s['label'],
        }
        rand_info.append(ss)
      print(ss)
    df_rinfo = pd.DataFrame(rand_info)
  return df_rinfo


def get_stack_df(train_files):
  dft = {s: pd.read_csv(f+'.csv') for s, f in train_files}
  dft_all = None
  dft_init = True
  for s, df in dft.items():
    df.rename(columns={'sponsored': s}, inplace=True)
    if dft_init:
      dft_init = False
      dft_all = df
    else:
      dft_all = pd.merge(dft_all, df, on='file', how='right')            
  # remove empty samples (where prediction = -1)
  dft_all = dft_all[dft_all['xgb']!=-1]
  return dft_all


def grade_predict(X, Y, model, en_plot=False):
  y2p = [[y, model.predict_proba(X[idx])[0][1]] for idx, y in enumerate(Y)]
  grd = grader.grader(en_plot=en_plot)
  auc = grd.auc_curve(y2p)
  return y2p


def train_stacking_model(train_samples, test_samples, fnames):
  # [samples]
  X_tr = train_samples.as_matrix(columns=fnames) #['ffm', 'xgb', 'slr', 'slr2'])
  Y_tr = list(train_samples['ans'])
  print("train shape: X_tr=%s / Y_tr=%s" % (np.shape(X_tr), np.shape(Y_tr)))
  X_te = test_samples.as_matrix(columns=fnames) #['ffm', 'xgb', 'slr', 'slr2'])
  Y_te = list(test_samples['ans'])
  print("test shape: X_te=%s / Y_te=%s" % (np.shape(X_te), np.shape(Y_te)))

  # [train stacking model]
  gs_params = {
      'C': np.linspace(0.001, 0.1, 4),
  }
  learner = linear_model.LogisticRegression()
  gs = grid_search.GridSearchCV(learner, gs_params, cv=4)
  gs.fit(X_tr, Y_tr)
  print("Model weights: ", gs.best_estimator_.coef_)

  # [grade]
  dmy = grade_predict(X_tr, Y_tr, gs, en_plot=True)
  dmy = grade_predict(X_te, Y_te, gs, en_plot=True)
  return gs


def check_cols(df, y2p_colname=[]):
  y2p = df.as_matrix(columns=y2p_colname)
  grd = grader.grader(en_plot=True)
  mauc = grd.auc_curve(y2p)
  return y2p


def p2rank(series, skey):
  sorted_series = sorted(zip(series[skey].values, series['file'].values))
  a = [{'file': v[1], skey+'_rank': idx} for idx,v in enumerate(sorted_series)]
  a_pd = pd.DataFrame(a)
  # a_pd.rename(columns={'rank': skey+'_rank'}, inplace=True)
  return a_pd


#==========================================
#   main flow
#==========================================
def go_stacking(files, sample_rand, model_weight=None, skip_basic=False):
  fnames = [n for n,f in files if n != 'ans']
  
  dft_train = get_stack_df(files)
  dft_all = pd.merge(dft_train, sample_rand, on='file', how='right')
  dft_all.drop('dans', axis=1, inplace=True)

  train_samples = dft_all[dft_all['rand'] <= 0.95]
  test_samples = dft_all[dft_all['rand'] > 0.95]
  print("%i train/%i test samples" % (train_samples.count().max(), test_samples.count().max()))
  
  if model_weight:
    gs = None
    test_samples['avg'] = sum([test_samples[fn]*w for fn, w in zip(fnames, model_weight)])/sum(model_weight)
  else:
    gs = train_stacking_model(train_samples, test_samples, fnames)
    test_samples['avg'] = test_samples[fnames].mean(axis=1)
  test_samples['med'] = test_samples[fnames].median(axis=1)
  for sm in (['avg'] if skip_basic else (['avg', 'med'] + fnames)):
    if sm == 'avg':
      print("check_cols auc: [%s], model_weight=%s" % (sm, model_weight))
    else:
      print("check_cols auc: [%s]" % (sm))
    check_cols(test_samples, y2p_colname=['ans', sm])
  return gs, train_samples, test_samples

def blending_by_rank_avg(test_samples):
  test_samples_rank = test_samples.merge(p2rank(test_samples, 'ffm'), on='file')
  test_samples_rank = test_samples_rank.merge(p2rank(test_samples, 'xgb'), on='file')
  test_samples_rank = test_samples_rank.merge(p2rank(test_samples, 'slr'), on='file')

  test_samples_rank['rank_avg'] = test_samples_rank[['ffm_rank', 'xgb_rank', 'slr_rank']].mean(axis=1)
  test_samples_rank = test_samples_rank.merge(p2rank(test_samples_rank, 'rank_avg'), on='file')
  test_samples_rank['rank_avg_rank'] = test_samples_rank['rank_avg_rank'] / test_samples_rank['rank_avg_rank'].max()

  print("check_cols auc: [rank_avg_rank]")
  check_cols(test_samples, y2p_colname=['ans', 'rank_avg_rank'])
  return test_samples_rank

#==========================================
#   test main
#==========================================
if __name__ == '__main__':
  sample_rand = get_sample_rand('/home/marsan/workspace/tnative/blending/stacking/sample_rand.pkl')
  # [train & check stacking model]
  gs, train_samples, test_samples = go_stacking([
    ('ans', 'train'),
    ('xgb', 'xgboost_t90_v9_auc_954_20151013_0425_2015_1014_0710'),
    ('slr', 'sklr_t90_v9_auc_948_20151011_2336_2015_1014_0721'),
    ('ffm', 'ffm_t90_v9_auc_947_20151012_0046_2015_1014_0711'),
    ('slr2', 'sklr_t95_v5_auc_929_20151014_2018_2015_1014_2036'),
  ], sample_rand, model_weight)

  # [try blending by rank average]
  test_samples_rank = blending_by_rank_avg(test_samples)
  
  # [generate submition]
  dft_submit = get_stack_df([
    ('xgb', 'xgboost_t90_v9_auc_954_20151013_0425_2015_1013_0610'),
    ('slr', 'sklr_t90_v9_auc_948_20151011_2336_2015_1012_0104'),
    ('ffm', 'ffm_t90_v9_auc_947_20151012_0046_2015_1012_0103'),
  ])

