import sys, random, json, os, re, pickle, hashlib, shutil, random
import numpy as np
import pandas as pd
pd.set_option('display.mpl_style', 'default')

import multiprocessing as mp
pool_size = int(mp.cpu_count())

from datetime import datetime
from bs4 import BeautifulSoup as bs
from mongoengine import Q
from nltk.stem.porter import *
from nltk.corpus import stopwords
from math import exp, log, sqrt, floor

root_path = '/home/marsan/workspace/tnative'
sys.path.append(root_path)
from lib import schema as db
# from lib import ner as ner


#==========================================
#   flow class
#==========================================  
class data_gen(object):
  def __init__(self, D=2**24, submit=False, interaction=False, en_fast_data=False, en_fast_load=False, layer2_ensemble=None,
                feature_select=[], feature_drop=[], remove_blacklist=False, rand_feats=False, debug=False,
              ):
    # params
    self.D = D
    self.en_fast_data = en_fast_data
    self.en_fast_load = en_fast_load
    self.interaction = interaction
    self.layer2_ensemble = layer2_ensemble
    if self.layer2_ensemble:
      self.booster_results = pickle.load(open(root_path+'/data/%s' % self.layer2_ensemble, 'rb'))
      print("layer2_ensemble: %s loaded @ %s" % (self.layer2_ensemble, datetime.now()))

    # ctrl
    self.feature_select = feature_select
    self.feature_drop = feature_drop
    self.remove_blacklist = remove_blacklist
    self.rand_feats = rand_feats
    self.submit = submit
    self.debug = debug

    # module
    # self.ner_tagger = ner.ner
    self.stemmer = PorterStemmer()

    # [teams feats_set]
    # self.en_feats_307 = True
    if not self.en_fast_data:
      tsvd_file = root_path+'/models/reduce_tfidf_dim.TruncatedSVD_50.model'
      self.tfidf_svd_model = pickle.load(open(tsvd_file, 'rb')) if os.path.isfile(tsvd_file) else None
      self.feats_307f = self.load_307_feats() # if self.en_feats_307 else {}
    else:
      self.tfidf_svd_model = None
      self.feats_307f = {}
    # print(self.feats_307f)

    self.rfecv_sel = False
    if self.rfecv_sel:
      # rfecv = pickle.load(open(root_path+'/models/sklr_t10_v9_auc_923_20151009_1340.pkl', 'rb'))[0].learner
      rfecv = pickle.load(open(root_path+'/models/rfecv_D20_s90.pkl', 'rb'))[0].learner
      self.true_idx = set([k for k,v in enumerate(rfecv.support_) if v > 0])
      print("[rfecv selected]", len(self.true_idx))

    # sample select
    lname = 'isad'
    self.blacklist = self.get_blacklist() if self.remove_blacklist else None
    if self.submit:
      self.tbl = db.articles.objects(isad=None)
    else:
      self.tbl = db.articles.objects(isad__ne=None, status__nin=['empty'])

    # info cache
    self.fidx = None
    self.info_set() # show features & D setting
    
    if self.debug:
      print('[dgen]: debug mode enabled, limit sample number to 100.')
      self.tbl = self.tbl.limit(100)


  #-------------------------
  #   text process
  #-------------------------
  def load_307_feats(self):
    def load_feats_set(filename, trim_cols=[]):
      hdata = {}
      data = pd.read_csv(filename)
      data['fid'] = [v.split('_')[0] for v in data['file']]
      for k in trim_cols: del data[k]
      for idx, d in data.iterrows():
        hdata[str(d.fid)] = list(d.values)[:-1]
        # if(idx > 10): break
      return hdata
    print("[load_307_feats] start @ %s" % datetime.now())
    trim_cols=['brackets', 'length', 'lines', 'spaces', 'tabs', 'words', 'file', 'sponsored']
    hdata_tr = load_feats_set(filename = root_path+'/data/kteam/train-features.csv', trim_cols=trim_cols)
    hdata_te = load_feats_set(filename = root_path+'/data/kteam/test-features.csv', trim_cols=trim_cols)
    hdata = dict(hdata_tr, **hdata_te)
    print("[load_307_feats] end @ %s" % datetime.now())
    return hdata

  #-------------------------
  #   text process
  #-------------------------
  def tag_link(self, link):
    if not link: return ''
    if '//' in link: link = link.split('//')[1]
    if '?' in link: link = link.split('?')[0]
    link = ' '.join(link.split('/')[0:3])
    return link

  def tag_text(self, text):
    if text == None: text = ''
    if isinstance(text, list): text = ' '.join([str(t) for t in text])
    text = re.sub('[^0-9a-zA-Z_-]+', ' ', text.lower())
    text = [self.stemmer.stem(t) for t in text.split()]
    text = list(set(text))                          
    return text

  def ensembler_dict(self, ens_dic):
    return ["%s_%s" % (k, int(v*20)) for k,v in ens_dic.items()]

  def str2hvec(self, s, D_clean=0):
    sbyte = str(s).lower().encode(encoding='UTF-8')
    shex = hashlib.md5(sbyte).hexdigest()
    return (int(shex, 36) % self.D) + D_clean

  def numerical2label(self, n): # no, not for 0~1 value
    return int(floor(log(abs(n or 1e-10))**2))
    # return log(abs(n or 1e-10))**2

  #-------------------------
  #   data blacklist
  #-------------------------
  def get_blacklist(self, filename=None):
    blacklist = {}
    # stop words
    print("get_blacklist start @ %s" % datetime.now())
    langs = ['english', 'french', 'german', 'spanish']
    stops = [j for k in [stopwords.words(lan) for lan in langs] for j in k]
    if filename:
      blacklist = pickle.load(open(fname, 'rb'))
    else:
      for atr in ['top_image', 'title', 'meta_site_name', 'meta_lang', 'meta_description', 'domain', 'canonical_link']:
      # for atr in ['title', 'meta_site_name', 'meta_lang', 'meta_description', 'domain']:
        fname = "%s/rares/%s.pkl" % (root_path, atr)
        if os.path.isfile(fname): 
          ritems = pickle.load(open(fname, 'rb'))
          # blacklist[atr] = list(set(ritems + stops))
          blacklist[atr] = set(ritems)
    print("get_blacklist done @ %s" % datetime.now())
    return blacklist

  #-------------------------
  #   db data to vectors
  #-------------------------
  def sample2vec(self, b, hashing=True, training=None):
    y = b.isad
    # [time]
    if (b.publish_date == None):
      label_month = label_wday = label_ihour = '-1'
    elif (b.publish_date < datetime.fromtimestamp(1e8)):
      label_month = label_wday = label_ihour = '-1'
    else:
      label_month = str(b.publish_date.month)
      label_wday  = str(b.publish_date.isoweekday())
      label_ihour = str(b.publish_date.hour)
    # [raw]
    x_raw = {
        'dummy'                   : '0',  # dummy for keep dimention while feature select                           
        'status'                  : str(b.status),
        'title'                   : self.tag_text(b.title),
        'tags'                    : self.tag_text(b.tags),                                      # drop: -0.01 /
        'keywords'                : self.tag_text(b.keywords),                                  # drop: -0.01 /
        'top_image'               : self.tag_text(b.top_image),
        'meta_site_name'          : self.tag_text(b.meta_site_name),
        'meta_lang'               : self.tag_text(b.meta_lang),
        'meta_description'        : self.tag_text(b.meta_description),
        'meta_keywords'           : self.tag_text(b.meta_keywords),                             # drop: -0.01 /
        'canonical_link'          : self.tag_text(self.tag_link(b.canonical_link)),             # drop: -0.01 /
        'domain'                  : self.tag_text(self.tag_link(b.domain)),
        # # # [author stat] good results! auc += 0.02
        'is_pure_ad_authors'      : str(b.is_pure_ad_authors == True),
        'is_pure_nonad_authors'   : str(b.is_pure_nonad_authors == True),
        'is_prefer_ad_authors'    : str(b.is_prefer_ad_authors == True),
        'is_prefer_nonad_authors' : str(b.is_prefer_nonad_authors == True),
        # # [domain stat]
        'is_pure_ad_domains'      : str(b.is_pure_ad_domains == True),
        'is_prefer_ad_domains'    : str(b.is_prefer_ad_domains == True),
        'is_prefer_nonad_domains' : str(b.is_prefer_nonad_domains == True),
        'is_pure_nonad_domains'   : str(b.is_pure_nonad_domains == True),
        # [numerical to label]
        'lang_adrate_group'       : str(b.lang_adrate_group),
        # [facebook stat]
        'fb_click_count'          : self.numerical2label(b.fb_click_count),
        'fb_like_count'           : self.numerical2label(b.fb_like_count),
        # [html inspections]
        'spaces'                  : self.numerical2label(b.spaces),
        'brackets'                : self.numerical2label(b.brackets),
        'quesmarks'               : self.numerical2label(b.quesmarks),
        'exclamarks'              : self.numerical2label(b.exclamarks),
        'words'                   : self.numerical2label(b.words),
        #
        'cnt_a'                   : self.numerical2label(b.cnt_a         ), #/96,
        'cnt_article'             : self.numerical2label(b.cnt_article   ), #/59,
        'cnt_b'                   : self.numerical2label(b.cnt_b         ), #/82,
        'cnt_blockquote'          : self.numerical2label(b.cnt_blockquote), #/47,
        'cnt_em'                  : self.numerical2label(b.cnt_em        ), #/86,
        'cnt_form'                : self.numerical2label(b.cnt_form      ), #/42,
        'cnt_h1'                  : self.numerical2label(b.cnt_h1        ), #/41,
        'cnt_h2'                  : self.numerical2label(b.cnt_h2        ), #/55,
        'cnt_h3'                  : self.numerical2label(b.cnt_h3        ), #/65,
        'cnt_h4'                  : self.numerical2label(b.cnt_h4        ), #/63,
        'cnt_h5'                  : self.numerical2label(b.cnt_h5        ), #/54,
        'cnt_h6'                  : self.numerical2label(b.cnt_h6        ), #/38,
        'cnt_iframe'              : self.numerical2label(b.cnt_iframe    ), #/30,
        'cnt_li'                  : self.numerical2label(b.cnt_li        ), #/85,
        'cnt_ol'                  : self.numerical2label(b.cnt_ol        ), #/28,
        'cnt_p'                   : self.numerical2label(b.cnt_p         ), #/110,
        'cnt_section'             : self.numerical2label(b.cnt_section   ), #/61,
        'cnt_select'              : self.numerical2label(b.cnt_select    ), #/31,
        'cnt_small'               : self.numerical2label(b.cnt_small     ), #/48,
        'cnt_strong'              : self.numerical2label(b.cnt_strong    ), #/71,
        'cnt_table'               : self.numerical2label(b.cnt_table     ), #/65,
        'cnt_textarea'            : self.numerical2label(b.cnt_textarea  ), #/30,
        'cnt_ul'                  : self.numerical2label(b.cnt_ul        ), #/70,
        #
        'cnt_at'                  : self.numerical2label(b.cnt_at        ), #/96,
        # 'cnt_blog'                : self.numerical2label(b.cnt_blog      ),
        'cnt_jpg'                 : self.numerical2label(b.cnt_jpg       ), #/112,
        'cnt_admin'               : self.numerical2label(b.cnt_admin     ), #/46,
        'cnt_click'               : self.numerical2label(b.cnt_click     ), #/76,
        'cnt_feed'                : self.numerical2label(b.cnt_feed      ), #/72,
        #
        'rto_html2txt'            : self.numerical2label(b.rto_html2txt  ), #/47,
        'cnt_meta'                : self.numerical2label(b.cnt_meta      ), #/88,
        # 'cnt_script'              : self.numerical2label(b.cnt_script    ),
        # 'cnt_style'               : self.numerical2label(b.cnt_style     ),
        # 'cnt_arrow'               : self.numerical2label(b.cnt_arrow     ),
        # 'cnt_slash'               : self.numerical2label(b.cnt_slash     ),
        # 'cnt_backslash'           : self.numerical2label(b.cnt_backslash ),
        # 'cnt_wp_content'          : self.numerical2label(b.cnt_wp_content),
        # 'cnt_dicts'               : 
        # [nlp]
        # 'opengraph'               : (b.opengraph or {}),
        # 'tfidf_vec'               : {k: n for k, n in enumerate(b.tfidf_vec) if abs(n) > 0},    # drop: +0.002
        # 'tfidf_vec_svd'           : {k: int(20*(n+1)) for k, n in enumerate(self.tfidf_svd_model.transform(b.tfidf_vec)[0])} if self.tfidf_svd_model else {}, 
        # 'doc2vec'                 : {k: int(20*(n+1)) for k, n in enumerate(b.doc2vec) if abs(n) > 0},
        'tfidf_vec_svd'           : {k: n for k, n in enumerate(self.tfidf_svd_model.transform(b.tfidf_vec)[0])} if self.tfidf_svd_model else {}, 
        'doc2vec'                 : {k: n for k, n in enumerate(b.doc2vec) if abs(n) > 0},
        'feats_307'               : {k: self.numerical2label(n) for k, n in enumerate(self.feats_307f.get(str(b.fid)) or {}) if abs(n) > 0} if self.feats_307f else {},
        'cnt_dicts'               : {k: self.numerical2label(n) for k, n in b.cnt_dicts.items() if abs(n) > 0},
        'cnt_bows'                : {k: self.numerical2label(n) for k, n in b.cnt_bows.items() if abs(n) > 0},
        # 'cnt_dicts_cnt'           : self.numerical2label(sum(b.cnt_dicts.values())),
        # 'cnt_bows_cnt'            : self.numerical2label(sum(b.cnt_bows.values())),
        # 'cleaned_text'            : self.tag_text(b.cleaned_text or b.html),                            # drop: -0.02
        #-----[team feats set]-----
        # 'rand_lognormal'          : {k: v for k,v in enumerate(np.random.lognormal(0, 1, 2))},
        # 'rand_normal'             : {k: v for k,v in enumerate(np.random.normal(0, 1, 2))},
        # 'rand_uniform'            : {k: v for k,v in enumerate(np.random.uniform(0, 1, 2))},
        #-----[feature selection: remove decicively]-----
        # 'fid'                     : self.numerical2label(b.fid),  # possible leak???
        # 'braces'                  : self.numerical2label(b.braces),
        # # [link stat]
        # 'has_pure_ad_links'       : str(b.has_pure_ad_links == True),
        # 'has_prefer_ad_links'     : str(b.has_prefer_ad_links == True),
        # 'has_pure_nad_links'      : str(b.has_pure_nad_links == True),
        # 'has_prefer_nad_links'    : str(b.has_prefer_nad_links == True),
        # ### 'summary'                 : self.tag_text(b.summary),                                   # drop: +0.008/
        # 'lines'                   : self.numerical2label(b.lines),
        # 'cnt_strike'              : self.numerical2label(b.cnt_strike    ),
        # 'cnt_img'                 : self.numerical2label(b.cnt_img       ),
        # 'cnt_input'               : self.numerical2label(b.cnt_input     ),
        # 'cnt_hr'                  : self.numerical2label(b.cnt_hr        ),
        # 'cnt_video'               : self.numerical2label(b.cnt_video     ),
        # 'cnt_code'                : self.numerical2label(b.cnt_code      ),
        #-----[feature droping: may try remove]-----
        # 'authors'                 : self.tag_text([a.replace(' ', '_') for a in b.authors]),    # drop: -0.01 /
        # 'fb_comment_count'        : self.numerical2label(b.fb_comment_count),
        # 'fb_share_count'          : self.numerical2label(b.fb_share_count),
        # 'fb_total_count'          : self.numerical2label(b.fb_total_count),
        # 'movies'                  : self.tag_text(b.movies),                                    # drop: +0.003/
        # 'tabs'                    : self.numerical2label(b.tabs),
        # 'links'                   : self.tag_text([self.tag_link(n) for n in (b.links or [])]),
        # 'html_cnt'                : self.numerical2label(b.html_cnt),
        # 'month'                   : label_month,                                                # drop: +0.004/
        # 'wday'                    : label_wday,                                                 # drop: +0.003/
        # 'ihour'                   : label_ihour,                                                # drop: +0.003/
    }
    if False: # self.layer2_ensemble:
      x_layer2 = {
        'booster'                   : self.ensembler_dict({k: v[b.fid] for k,v in self.booster_results.items()}),
        # 'ensemble_0'              : self.ensembler_dict(b.ensemble_0     ), 
        # 'ensemble_1'              : self.ensembler_dict(b.ensemble_1     ),
        # 'ensemble_2'              : self.ensembler_dict(b.ensemble_2     ),
        # 'ensemble_3'              : self.ensembler_dict(b.ensemble_3     ),
        # 'ensemble_4'              : self.ensembler_dict(b.ensemble_4     ),
      }
      x_raw = dict(x_raw, **x_layer2)
    
    # [feature hashing]
    if hashing: 
      x = self.hashing_raw(x_raw, hashing=hashing)
    else:
      x = x_raw
    return y, x


  def prime_vals(self, v):
    if isinstance(v, bool):
        vs = 1 if v else 1e-10
    elif isinstance(v, str):
      if v == 'True':
        vs = 1
      elif v == 'False':
        vs = 1e-10
      elif v in ['None', '']:
        vs = -1e-10
      elif v.replace(".", "", 1).isdigit():
        vs = float(v)
      else:
        print("[EXCEPTION] in prime_vals: vs = %s" % vs)
        vs = -1e-10
    elif v in [530, None]:
      vs = -1e-10
    else:
      vs = v
    # print(v, vs)
    return vs

  def hashing_raw(self, x_raw, hashing=True):
    # random features as baseline of feature selection
    # if self.rand_feats:
    #   x_rand = {
    #     'rand_lognormal'  : {k: v for k,v in enumerate(np.random.lognormal(0, 1, 2))},
    #     'rand_normal'     : {k: v for k,v in enumerate(np.random.normal(0, 1, 2))},
    #     'rand_uniform'    : {k: v for k,v in enumerate(np.random.uniform(0, 1, 2))},
    #   }
    #   x_raw = dict(x_raw, **x_raw)

    # [feature select]
    if self.feature_select:
      single_features = [s for s in self.feature_select if '|' not in s]
      x_raw_sel = {k: v for k,v in x_raw.items() if k in single_features}
      for it in [s for s in self.feature_select if '|' in s]:
        x_raw_sel[it] = "_".join([str(x_raw[k]) for k in it.split('|')])
      x_raw = x_raw_sel
    if self.feature_drop:
      for d in self.feature_drop:
        del x_raw[d]

    # [known feature drop]
    # minors = [
    #   'words', 'cnt_jpg', 'cnt_textarea', 'cnt_at', 
    #   'cnt_select', 'cnt_form', 'cnt_ol',
    #   'cnt_meta', 'fb_like_count', 'cnt_em', 'meta_keywords',
    # ]
    too_strong = [
      'is_pure_ad_domains', 'is_prefer_ad_domains', 'is_prefer_nonad_domains', 'is_pure_nonad_domains',
      'is_pure_ad_authors', 'is_prefer_ad_authors', 'is_prefer_nonad_authors', 'is_pure_nonad_authors',
    ]
    for m in too_strong:
      if m in x_raw:
        del x_raw[m]

    # [remove blacklist]  
    if self.remove_blacklist: # & (y if training else True)): #(NOTE: modify only True samples, but we don't know y while testing)
      for c in self.blacklist.keys():
        x_raw[c] = [t for t in x_raw[c] if ((t not in self.blacklist[c]) & (len(t) > 3))]


    # [hashing & output stage]
    x = self.hashing_obj(x_raw)
    
    # rfecv selected features
    if self.rfecv_sel:
      x = [(k, hval, val) for k, hval, val in x if self.str2hvec("%s_%s" % (self.fidx[k], hval)) in self.true_idx]

    # final hashing
    primes = [
      # dynamic lists
      # 'tags', 'domain', 'cnt_dicts', 'cnt_bows', 'feats_307', 'title', 'meta_lang', 'keywords',
      # 'meta_keywords', 'meta_description', 'canonical_link'
      # fixed length list
      # 'tfidf_vec_svd', 'doc2vec', 'meta_site_name', 'status',
      # single
      'dummy', 'rto_html2txt', 'cnt_small', 'cnt_admin', 'cnt_article',
      'cnt_li', 'is_prefer_nonad_domains', 'cnt_strong', 'cnt_blockquote', 
      'words', 'is_pure_nonad_domains', 'cnt_p', 'is_prefer_nonad_authors', 
      'fb_click_count', 'cnt_ul', 'fb_like_count', 'cnt_jpg', 'cnt_a', 'cnt_click', 
      'is_pure_nonad_authors', 'cnt_h6', 'cnt_meta', 'cnt_h5', 'cnt_b', 'cnt_em', 'brackets', 
      'cnt_ol', 'quesmarks', 'is_pure_ad_domains', 'is_prefer_ad_domains', 'is_pure_ad_authors', 'is_prefer_ad_authors', 
      'lang_adrate_group', 'cnt_table', 'cnt_h3', 'cnt_feed', 'cnt_textarea', 
      'spaces', 'cnt_h1', 'cnt_iframe', 'cnt_at', 'cnt_h4', 
      'cnt_h2', 'cnt_select', 'cnt_form', 'exclamarks', 'cnt_section', 
    ]
    x_primes = [self.prime_vals(x_raw[p]) for p in primes if p in x_raw] 
    for klist in ['tfidf_vec_svd', 'doc2vec']:
      # if x_raw.get(klist): 
      x_primes += list(x_raw[klist].values())
    primes_os = len(x_primes)
    # print("primes_os=%i" % primes_os)
    
    if (hashing == 'ffm'): 
      x_primes = [(idx, self.str2hvec(idx), val) for idx, val in enumerate(x_primes)] 
    else:
      x_primes = [(idx, val) for idx, val in enumerate(x_primes)] 

    # if (hashing == 'ffm'):
    #   x_res = [(self.fidx[k], self.str2hvec(hval), val) for k, hval, val in x]
    # else:
    #   x_res = [(self.str2hvec("%s_%s" % (self.fidx[k], hval)), val) for k, hval, val in x]
    if (hashing == 'ffm'):
      x_res = [(self.fidx[k]+300, self.str2hvec(hval, D_clean=10000), val) for k, hval, val in x if k not in x_primes]
    else:
      x_res = [(self.str2hvec("%s_%s" % (self.fidx[k], hval), D_clean=10000), val) for k, hval, val in x if k not in x_primes]

    return x_primes + x_res


  def hashing_obj(self, x_raw):
    x = []
    for k, v in x_raw.items():  # feature hashing
      try:
        if isinstance(v, list):
          for idx, vv in enumerate(v):
            if isinstance(vv, float):
              s = (k, str(idx), vv)
            else:
              s = (k, str(vv), 1)
            x.append(s)
        elif isinstance(v, dict):
          for kk, vv in v.items():
            if isinstance(vv, float):
              s = (k, str(kk), vv)
            else:
              s = (k, ("%s_%s" % (kk, vv)), 1)
            x.append(s)
        else:  # str or unicode
          if isinstance(v, float):
            s = (k, None, v)
          else:
            s = (k, str(v), 1)
          x.append(s)
      except Exception as e:
        print("[ERROR]%s/%s/%s" % (k, v, e))

    x = list(set(x))
    return x


  def interact_x(self, x):
    x_int = []
    D = self.D
    L = len(x)
    x = sorted(x)
    for i in range(L):
      for j in range(i+1, L):
        xi = abs(hash(str(x[i]) + '_' + str(x[j]))) % D
        x_int.append(xi)
    return list(set(x + x_int))

  #-------------------------
  #   generator
  #-------------------------
  def raw_range(self, rmin, rmax):
    if rmax > rmin:
      raw = self.tbl.filter(rand__gte=rmin, rand__lt=rmax)
    else:
      raw = self.tbl.filter(Q(rand__gte=rmin) | Q(rand__lt=rmax))
    raw = raw.exclude('html', 'cleaned_text') # save memory
    return raw.order_by('id')


  def gen_data(self, rmin, rmax, hashing=True, extra=False):
    if self.en_fast_load:
      print("[gen data] through fast_load @ %s" % datetime.now())
      stamp = 'fast_gen_tmp_%i' % int(random.random()*1e6)
      self.en_fast_data = gen_fast_data(stamp, D=self.D, rmin=rmin, rmax=rmax, submit=self.submit, hashing=hashing)
      if '/' in self.en_fast_data: self.en_fast_data = self.en_fast_data.split('/')[-1]
    if self.en_fast_data:
      print("[gen data] through fast_data %s, start @ %s" % (self.en_fast_data, datetime.now()))
      pkl_folder = "%s/data/%s" % (root_path, self.en_fast_data)
      files = [ f for f in os.listdir(pkl_folder) if os.path.isfile(os.path.join(pkl_folder, f)) ]
      idx = 0
      for f in files:
        samples = pickle.load(open("%s/%s" % (pkl_folder, f), 'rb'))
        for s in samples:
          if (((s['rand'] >= rmin) & (s['rand'] < rmax)) | self.en_fast_load):
            # if isinstance(s['features'], dict): 
            s['features'] = self.hashing_raw(s['features'], hashing=hashing)
            if extra:
              yield idx, s['features'], s['label'], {'fid': s['fid'], 'rand': s['rand']}
            else:
              yield idx, s['features'], s['label']
            idx += 1
      print("gen_data completed for %i samples @ %s" % (idx, datetime.now()))
      if self.en_fast_load: shutil.rmtree(pkl_folder)  # clear tmp folders
    else:
      print("[gen data] through SQL, start @ %s" % (datetime.now()))
      if not self.debug: self.observe_data(rmin, rmax)
      raw = self.raw_range(rmin, rmax).order_by('id').timeout(False).no_cache()
      for idx, r in enumerate(raw):
        y, x = self.sample2vec(r, hashing=hashing)
        if self.debug: y = idx % 2  # ensure both 0 & 1 samples exists.
        if self.interaction: x = self.interact_x(x)
        if extra:
          yield idx, x, y, {'fid': r.fid, 'rand': r.rand}
        else:
          yield idx, x, y

  #-------------------------
  #   observer
  #-------------------------
  def info_set(self): # get a unique hash number of all feature names & D
    # if self.en_fast_data:
    #   return self.en_fast_data
    # else:
      feats = sorted(list(self.rand_sample()[1].keys()) + [str(self.D)])
      self.fidx = {fea: idx for idx, fea in enumerate(feats)}
      # print("dgen fidx = %s" % self.fidx)
      s = "_".join(feats)
      # print("info_set: %s" % s)
      return self.str2hvec(s)

  def rand_sample(self, sid=None, hashing=False):
    t1 = self.tbl.filter(id=sid).first() if sid else self.tbl[int(1000*random.random())]
    t1_vec = self.sample2vec(t1, hashing=hashing)
    return t1_vec


  def observe_data(self, rmin, rmax, silent=False):
    raw = self.raw_range(rmin, rmax)
    ad_cnt = raw.filter(isad=True).count()
    nad_cnt = raw.filter(isad=False).count()
    all_cnt = raw.count() if self.submit else ad_cnt + nad_cnt
    if not all_cnt: all_cnt = -1   # prevent division by 0
    if not silent: print("[data] %i to go, ad_rate=%i/%i=%.3f%%" % (all_cnt, ad_cnt, nad_cnt, 100*float(ad_cnt)/all_cnt))
    return ad_cnt, nad_cnt, all_cnt
    

#==========================================
#   pickle fast samples
#==========================================
def pkl_samples(mode_name, D, rmin, rmax, submit=False, debug=False, remove_blacklist=False, layer2_ensemble=None, hashing=True):
  items = []
  # NOTE: 'en_fast_load' CANNOT be true here, will cause 'children thread of thread'.
  dg = data_gen(D=D, en_fast_data=False, en_fast_load=False, submit=submit, remove_blacklist=remove_blacklist, layer2_ensemble=layer2_ensemble) 
  for idx, x, y, extra in dg.gen_data(rmin, rmax, extra=True, hashing=hashing):
    item = {
      'label'     : y,
      'features'  : set(x) if isinstance(x, list) else x,
      'rand'      : extra['rand'],
      'fid'       : extra['fid'],
    }
    items.append(item)
    if (idx % 10000 == 0): print("%s - %s: %i done @ %s" % (rmin, rmax, idx, datetime.now()))
    if (debug & (idx > 3000)): break
  fname = root_path+"/data/%s/%s_%s_%s.pkl" % (mode_name, mode_name, int(10000*rmin), int(10000*rmax))
  pickle.dump(items, open(fname, 'wb'))
  return #fname


def gen_fast_data(mode_name, D=2**24, rmin=0, rmax=1, booster=None, submit=False, bsize=10, debug=False, remove_blacklist=False, layer2_ensemble=None, hashing=True):
  # clear folder
  newpath = "%s/data/%s" % (root_path, mode_name)
  if os.path.exists(newpath): shutil.rmtree(newpath)
  os.makedirs(newpath)
  # gen data
  mp_pool = mp.Pool(pool_size*2)
  processes = []
  for i in np.linspace(rmin, rmax, bsize+1)[1:]:
    p = mp_pool.apply_async(pkl_samples, (mode_name, D, i - (rmax-rmin)/bsize, i, submit, debug, remove_blacklist, layer2_ensemble, hashing))
    processes.append(p)
  mp_pool.close()
  mp_pool.join()
  for idx, p in enumerate(processes): 
    p.get()
    # print("process completed: %i / %s" % (idx, datetime.now()))
  return newpath
