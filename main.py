import sys, gzip, json, os, math, pickle, re
import numpy as np
from datetime import datetime
from math import exp, log, sqrt


root_path = '/home/marsan/workspace/tnative'
sys.path.append(root_path)
from lib import dgen as dgen
from lib import top as top

#==========================================
#   Tuner
#==========================================
class model_tuner(object):
  def __init__(self, D=2**20, en_plot=False, save=False, en_fast_data=False, en_fast_load=False, interaction=False):
    self.D = D
    self.en_plot = en_plot
    self.save = save
    self.en_fast_data = en_fast_data
    self.en_fast_load = en_fast_load
    self.interaction = interaction

    # feature select
    self.feats_to_check = [
      # 'dummy', 
      # ['is_pure_ad_domains', 'is_pure_nonad_domains', 'is_pure_ad_authors', 'is_pure_nonad_authors', 'is_prefer_ad_domains', 'is_prefer_ad_authors'],
      # ['fb_click_count'],
      # 'keywords', 'meta_description', 'title', 'top_image', 'tags',
      ['brackets', 'canonical_link', 'domain', 'is_pure_nonad_domains', 'meta_description', 
        'title', 'meta_site_name', 'top_image', 'cnt_dicts'],
      ['brackets', 'canonical_link', 'domain', 'is_pure_nonad_domains', 'meta_description', 
        'title', 'meta_site_name', 'top_image', 'cnt_dicts', 'meta_lang', 'exclamarks', 
        'cnt_ul', 'cnt_a', 'cnt_p', 'cnt_li', 'cnt_table', 'cnt_section', 'cnt_h3'],
      ['brackets', 'canonical_link', 'domain', 'is_pure_nonad_domains', 'meta_description', 
        'title', 'meta_site_name', 'top_image', 'cnt_dicts', 'meta_lang', 'exclamarks', 
        'cnt_ul', 'cnt_a', 'cnt_p', 'cnt_li', 'cnt_table', 'cnt_section', 'cnt_h3', 'cnt_dicts', 
        'cnt_bows', 'tfidf_vec_svd', 'doc2vec', 'feats_307', 'spaces', 'keywords', 'cnt_table', 'fb_like_count'],
      # 'rto_html2txt', 'cnt_script', 'cnt_style', 'cnt_arrow', 'cnt_slash', 'cnt_backslash', 'cnt_meta', 'cnt_wp_content',
    ]
    self.ensured_feats = set([
      # 'tfidf_vec', 
      # 'wday', 'month', 'status',
      # 'cnt_at', 'cnt_blog', 'cnt_jpg', 'cnt_admin', 'cnt_click', 'cnt_feed', 
      # 'brackets', 'canonical_link', 'domain', 'title', 'keywords',
      # 'is_pure_nonad_domains', 'is_pure_ad_domains', 'is_prefer_ad_domains', 'is_prefer_nonad_domains',
      # 'is_pure_ad_authors', 'is_pure_nonad_authors', 'is_prefer_ad_authors', 'is_prefer_nonad_authors',
      # 'lang_adrate_group', 'fb_click_count', 'fb_like_count',
      # 'top_image', 'meta_description', 'meta_site_name', 'meta_lang', 'exclamarks', 'quesmarks', 'words',
      # 'cnt_section', 'cnt_ul', 'cnt_li', 'cnt_article', 'cnt_a', 'cnt_p', 
      # 'cnt_select', 'cnt_blockquote', 'cnt_b', 'cnt_code', 'cnt_form', 
      # 'cnt_h1', 'cnt_h2', 'cnt_h3', 'cnt_h4', 'cnt_h5', 'cnt_h6', 'cnt_table', 'cnt_textarea',
      # 'cnt_ol', 'cnt_small', 'cnt_strong', 'html_cnt',
    ])

    # for model greedy search
    self.model_params = {
      'ffm': {
        'eta'     : [0.01, 0.05, 0.2], # smaller learning-rate (0.01) for big data, larger (> 0.1) for smaller data size.
        # 'lambd'   : [5e-7, 1e-6, 2e-6], #np.linspace(2e-6, 2e-5, 4), # larger trimming (2e-5) for big data, little trimming (2e-6) for smaller data size.
        # 'k'       : [2, 4, 8],      # > 4 tend to overfit, unless have more more data.
        # 'itera'   : [20, 25, 30],   # the more the better... maybe.
        # 'fold'    : [2, 4, 8],      # the more the better, but improve very little.
      },
      'xgboost': {
        'learning_rate': [0.1, 0.2, 0.3], # most critical even small change
        # 'max_depth': [7, 9, 12],            # larger = better or overfitting 
        # 'n_estimators': [300, 500, 700],  # larger = better = slower
      },
      'sklr': {
        'C'       : [0.02, 0.2], # [0.2, 0.25, 0.3, 0.35, 0.4], 
        # 'C'       : np.linspace(0.2, 0.3, 5),   # smaller learning-rate (0.01) for big data, larger (> 0.1) for smaller data size.
        # 'penalty' : ['l1', 'l2'],             # l1 only if you need sparse model
        # 'max_iter'  : np.logspace(2, 3, 4),   # the more the better... maybe.
      },
      'skgbdt': {
        'learning_rate' : 0.1,
        'n_estimators'  : 100,
      },
      'skrf': {
        'n_estimators'  : [100, 200, 300],
      },
      'ftrl': {
        # 'epoch' : list(range(1, 10)),
        'alpha' : np.logspace(-4, 0, 5),
        'beta'  : np.logspace(-4, 0, 5),
        'L1'    : np.logspace(-2, 2, 5),
        'L2'    : np.logspace(-2, 2, 5),
      },
      'sksgd': {
        # 'epoch'     : list(range(1, 10)),
        'alpha'     : np.logspace(-4, -1, 10),
        'l1_ratio'  : np.linspace(0, 1, 10),  # 0 <= l1_ratio <= 1
      }
    }
  
  #-----------------------------------
  #   Feature selection
  #-----------------------------------
  def get_feats(self):
    dummy_flow = top.ml(alg='sklr', D=self.D, en_fast_data=False)
    features = sorted(dummy_flow.dgen.rand_sample(hashing=False)[1].keys())
    if self.feats_to_check:
      features = self.feats_to_check #[f for f in features if f in self.feats_to_check]
    else:
      features = [f for f in features if f not in self.ensured_feats]
    return features

  def feature_select(self, alg='sklr', srate=0.1, interaction=False):
    results = []
    features = self.get_feats()
    print("[feature_select] for %s start @ %s" % (features, datetime.now()))
    if interaction:
      for f1 in features:
        for f2 in features:
          if (f1 == f2): break
          fea = "%s|%s" % (f1, f2)
          print("%s[%s]%s" % ('='*5, fea, '='*80))
          if self.en_fast_data:
            flow = top.ml(alg=alg, D=self.D, en_fast_data=self.en_fast_data, feature_select=['dummy', fea], save=self.save)
          else:
            flow = top.ml(alg=alg, D=self.D, en_fast_load=True, feature_select=['dummy', fea], save=self.save)
          if isinstance(srate, list):
            auc = flow.train(srate[0], srate[1], srate[2], srate[3])
          else:
            auc = flow.train(0, srate)
          results.append({'auc': auc, 'features': [fea]})
    else:
      for f1 in features:
        print("%s[%s]%s" % ('='*5, f1, '='*80))
        feats = ['dummy'] + f1 if isinstance(f1, list) else ['dummy', f1]
        if self.en_fast_data:
          flow = top.ml(alg=alg, D=self.D, en_fast_data=self.en_fast_data, feature_select=feats, save=self.save)
        else:
          flow = top.ml(alg=alg, D=self.D, en_fast_load=False, feature_select=feats, save=self.save)
        if isinstance(srate, list):
          auc = flow.train(srate[0], srate[1], srate[2], srate[3])
        else:
          auc = flow.train(0, srate)
        results.append({'auc': auc, 'features': feats})
    print("%s[Summary]%s" % ('='*5, '='*80))
    for r in results: print(r)

  def feature_drop(self, alg='sklr', srate=0.1):
    results = []
    features = self.get_feats()
    print("[feature_drop] for %s start @ %s" % (features, datetime.now()))
    for f1 in features:
      print("%s[drop %s]%s" % ('='*5, f1, '='*80))
      feats = f1 if isinstance(f1, list) else [f1]
      if self.en_fast_data:
        flow = top.ml(alg=alg, D=self.D, en_fast_data=self.en_fast_data, feature_drop=feats, save=self.save)
      else:
        flow = top.ml(alg=alg, D=self.D, en_fast_load=True, feature_drop=feats, save=self.save)
      if isinstance(srate, list):
        auc = flow.train(srate[0], srate[1], srate[2], srate[3])
      else:
        auc = flow.train(0, srate)
      results.append({'auc': auc, 'drops': [f1]})
    print("%s[Summary]%s" % ('='*5, '='*80))
    for r in results: print(r)


  #-----------------------------------
  #   Search parameters
  #-----------------------------------
  def compare_alg_srate(self, grid_srate=None, grid_alg=None):
    if not grid_srate: grid_srate = [0.01, 0.1, 0.3, 0.5, 0.8]
    if not grid_alg:   grid_alg = ['xgboost', 'sklr', 'ftrl'] #, 'skrf', 'sksgd', 'skgbdt']
    results = []
    for srate in grid_srate:
      for alg in grid_alg:
        flow = top.ml(alg=alg, D=self.D, en_plot=self.en_plot, save=self.save, en_fast_data=self.en_fast_data, interaction=self.interaction)
        auc = flow.train(0, srate)
        results.append({'auc': auc, 'alg': alg, 'srate': srate})
    print("%s[Summary]%s\n" % ('='*5, '='*80), results)
    
  def search_alg_params(self, alg, srate, key, values):
    print("%s\n#  [%s] grid search for %s in [%s], srate=%s\n%s" % ("="*60, alg, key, values, srate, "="*60))
    results = []
    for v in values:
      params = {key: v}
      print('self.D', self.D)
      flow = top.ml(alg=alg, D=self.D, params=params, en_plot=self.en_plot, save=self.save, en_fast_data=self.en_fast_data, en_fast_load=self.en_fast_load, interaction=self.interaction)
      if isinstance(srate, list):
        auc = flow.train(srate[0], srate[1], srate[2], srate[3])
      else:
        auc = flow.train(0, srate)
      results.append({'auc': auc, 'params': flow.learner.params})
    return results

  def greedy_search(self, model, srate=0.9):
    results = []
    for k, v in self.model_params[model].items():
      results += self.search_alg_params(model, srate, k, v)
    print("%s[Summary]%s" % ('='*5, '='*80))
    for r in results: print(r)

  
#==========================================
#   test main
#==========================================
if __name__ == '__main__':
  if (len(sys.argv) < 2): 
    print(''' 
      [Usage] 
        1. train single model with sample rate: 
          ipython main.py [alg: xgboost/sklr/ftrl/skrf/skgbdt/sksgd] [srate: 0.01 ~ 1.0] 
        2. compare all algs and sample rates
          ipython main.py compare_alg_srate
        3. greedy search model parameters
          ipython main.py greedy_search [alg]
        4. feature_select
          ipython main.py feature_select interaction
        5. feature_drop
          ipython main.py feature_drop
    ''')
    sys.exit()
  cmd = str(sys.argv[1])
  if (len(sys.argv) >= 3): cmd2 = str(sys.argv[2])
  if cmd == 'compare_alg_srate':
    model_tuner().compare_alg_srate()
  elif cmd == 'greedy_search':
    model_tuner(save=True, en_fast_load=True).greedy_search(cmd2, srate=[0, 0.9, 0.9, 1])
  elif cmd == 'fast_greedy_search':
    model_tuner(D=2**20, save=True, en_fast_data='D_20_all').greedy_search(cmd2, srate=[0, 0.95, 0.95, 1])
  elif cmd == 'feature_select':
    interaction = (cmd2 == 'interaction')
    print("feature_select: interaction=%s" % interaction)
    # model_tuner(save=False, en_fast_data='D_20_all').feature_select(alg='sklr', srate=[0, 0.1, 0.9, 1], interaction=interaction)
    model_tuner(D=2**20, save=True, en_fast_data='D_20_all').feature_select(alg='xgboost', srate=[0, 0.9, 0.9, 1], interaction=interaction)
  elif cmd == 'feature_rfecv':  # recursive feature elimination and cross-validated selection
    top.ml(alg='sklr', D=2**20, params={'rfecv': True}, save=True, en_fast_data='D_20_all').train(0, 0.3, 0.9, 1)
  elif cmd == 'feature_drop':
    model_tuner(save=False, en_fast_data='D_20_all').feature_drop(alg='sklr', srate=[0, 0.1, 0.9, 1])
  elif cmd == 'bagging_layer1':
    for rto in [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6]:
      top.ml(alg=cmd2,  en_plot=False, save=True, en_fast_data='D_20').train(rto, rto+0.3, 0.9, 1)
  elif cmd == 'booster_layer2':
    top.ml(alg=cmd2, en_plot=False, save=True, en_fast_data='D_20').train(0, 0.9, 0.9, 1)
  elif cmd == 'drop_exp':
    # compare
    print("comparison start @ %s" % datetime.now())
    flow = top.ml(alg='sklr', feature_drop=drop_items)
    flow.train(0, 0.1, 0.9, 1)
    # experimant
    print("experimant start @ %s" % datetime.now())
    drop_items = ['authors', 'fb_comment_count', 'fb_share_count', 'fb_total_count', 'movies', 'tabs', 'links']
    print("[drop_exp] drop %s start @ %s" % (drop_items, datetime.now()))
    flow = top.ml(alg='sklr', feature_drop=drop_items)
    flow.train(0, 0.1, 0.9, 1)
  else:
    srate = float(sys.argv[2])
    # top.ml(alg=cmd, en_plot=False, save=True, dump_y2p_ids=True).train(0, srate)
    top.ml(alg=cmd, en_plot=False, save=True, dump_y2p_ids=True).train(0, 0.9, 0.9, 1)
    # top.ml(alg=cmd, en_plot=False, save=True, en_fast_data='D_20_beta_rm_blacklist_w_stats').train(0, srate)
    
