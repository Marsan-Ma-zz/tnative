import sys, gzip, json, os, math, pickle, re, random
import scipy.sparse as sparse
import numpy as np
import multiprocessing as mp
pool_size = int(mp.cpu_count())

from datetime import datetime
from math import exp, log, sqrt
from bs4 import BeautifulSoup as bs
from collections import Counter, OrderedDict

root_path = '/home/marsan/workspace/tnative'
sys.path.append(root_path)

# models
from lib import schema as db
from lib import ensembler as ens
from lib import top as top


#==========================================
#   main class
#==========================================  
class prober(object):
  def __init__(self, alg='sklr', en_plot=False):
    self.alg = alg
    self.en_plot = en_plot


  #-------------------------
  #   compare
  #-------------------------
  def probe_fp_fn(self, srate=0.8):
    # get results
    fname = ('%s_%s' % (self.alg, int(srate*100)))
    top.ml(alg=self.alg, en_plot=self.en_plot, dump_y2p_ids=fname).train(0, srate)
    y2p, ids = pickle.load(open(fname+'.pkl', 'rb'))
    # collect
    th = 0.5
    tp = [ids[idx] for idx, (y, p) in enumerate(y2p) if ((p >= th) & (y > 0))]
    fp = [ids[idx] for idx, (y, p) in enumerate(y2p) if ((p >= th) & (y == 0))]
    tn = [ids[idx] for idx, (y, p) in enumerate(y2p) if ((p < th) & (y == 0))]
    fn = [ids[idx] for idx, (y, p) in enumerate(y2p) if ((p < th) & (y > 0))]
    print('tp', len(tp), 'fp', len(fp), 'tn', len(tn), 'fn', len(fn))
    db.articles.objects(id__in=fp).update(debug='%s_fp' % fname)
    db.articles.objects(id__in=fn).update(debug='%s_fn' % fname)
    print("debug = [%s, %s] saved" % ('%s_fp' % fname, '%s_fn' % fname))


  #-------------------------
  #   fast gather data
  #-------------------------
  def fast_gather_atrs(self, rmin=0, rmax=1, bsize=20, atr='bow', isad=None):
    mp_pool = mp.Pool(pool_size*2)
    processes = []
    for i in np.linspace(rmin, rmax, bsize+1)[1:]:
      params = (i - (rmax-rmin)/bsize, i, atr, isad) if (isad != None) else (i - (rmax-rmin)/bsize, i, atr) 
      p = mp_pool.apply_async(get_atrs, params)
      processes.append(p)
    mp_pool.close()
    mp_pool.join()
    all_bows = []
    for idx, p in enumerate(processes): 
      all_bows += p.get()
    print("[fast_gather_atrs] done with %i bows @ %s" % (len(all_bows), datetime.now()))
    return all_bows

  #-------------------------
  #   probe features
  #-------------------------
  def count_numerical_max(self, atrs):
    results = {}
    if not atrs:
      atrs = ['fb_click_count', 'fb_like_count', 'spaces', 'brackets', 'quesmarks', 'exclamarks', 'words', 'cnt_a', 'cnt_article', 'cnt_b', 'cnt_blockquote', 'cnt_em', 'cnt_form', 'cnt_h1', 'cnt_h2', 'cnt_h3', 'cnt_h4', 'cnt_h5', 'cnt_h6', 'cnt_iframe', 'cnt_li', 'cnt_ol', 'cnt_p', 'cnt_section', 'cnt_select', 'cnt_small', 'cnt_strong', 'cnt_table', 'cnt_textarea', 'cnt_ul', 'cnt_at', 'cnt_jpg', 'cnt_admin', 'cnt_click', 'cnt_feed', 'rto_html2txt', 'cnt_meta']
    for atr in atrs:
      r = db.articles.objects.only(atr).order_by('-'+atr).first()
      results[atr] = getattr(r, atr)
      print(atr, getattr(r, atr))
    return results

  #-------------------------
  #   probe html
  #-------------------------
  def collect_bows(self, sample_rto=0.01, uniq_cnt=False):
    ad_bows = self.fast_gather_atrs(rmax=sample_rto, atr='bow', isad=True)
    nad_bows = self.fast_gather_atrs(rmax=sample_rto, atr='bow', isad=False)
    if uniq_cnt:
      ad_bows = [j for fid, k in ad_bows for j in set(k)]
      nad_bows = [j for fid, k in nad_bows for j in set(k)]
    return ad_bows, nad_bows


  def collect_htmls(self, sample_cnt=1000):
    def rand_sample(isad):
      html = db.articles.objects(isad=True)[int(random.random()*1000)].html.lower()
      return html
    ad_htmls = []
    nad_htmls = []
    for i in range(sample_cnt):
      ad_htmls.append(rand_sample(isad=True)) 
      nad_htmls.append(rand_sample(isad=False)) 
      if (i % sample_cnt/10 == 0): print('collect_htmls: %i collected @ %s' % (i, datetime.now()))
    print('[collect_htmls] ad_htmls:', len(ad_htmls), 'nad_htmls:', len(nad_htmls))
    return ad_htmls, nad_htmls


  def probe_html_items(self, items, sample_cnt=1000):
    ad_htmls, nad_htmls = self.collect_htmls(sample_cnt)
    mp_pool = mp.Pool(pool_size*2)
    processes = []
    for key in items:
      # ad_res, nad_res = stat_feats(ad_htmls, nad_htmls, key=key)
      p = mp_pool.apply_async(stat_feats, (ad_htmls, nad_htmls, key))
      processes.append([key, p])
    for k, p in processes:
      ad_res, nad_res = p.get() #timeout=10)
    processes = []

#-------------------------
#   multi-task elements
#-------------------------
def get_atrs(pmin, pmax, atr='bow', isad=None):
  # print("get_atrs %s for %.6f - %.6f start @ %s" % (atr, pmin, pmax, datetime.now()))
  bows = []
  raw = db.articles.objects(rand__gte=pmin, rand__lt=pmax)
  if (isad != None): raw = raw.filter(isad=isad)
  raw = raw.only(atr).timeout(False).no_cache()
  for r in raw:
    bows.append([str(r.id), getattr(r, atr)])
  print("get_atrs %s for %.6f - %.6f done @ %s" % (atr, pmin, pmax, datetime.now()))
  return bows
  


def stat_feats(ad_htmls, nad_htmls, key):
  def html_feat(html):
    soup = bs(html)
    cnt = html.count(key.lower())
    return cnt
  ad_res = [html_feat(html) for html in ad_htmls]
  nad_res = [html_feat(html) for html in nad_htmls]
  print("[%s] SUM: ad_cnt=%i / nonad_cnt=%i @ %s" % (key, sum(ad_res), sum(nad_res), datetime.now()))
  return ad_res, nad_res
    

#==========================================
#   try
#==========================================  
if __name__ == '__main__':
  prb = prober(en_plot=False)
  # prb.probe_fp_fn(0.8)
  # items = [
  #     # 'X-UA-Compatible', 'og:image', 'fb:app_id', 'twitter', '.jpg', ':', '@', 'blog', 'author', 'admin', 'device',
  #     # 'click', 'feed', 'follow', 'like', 'yes',
  #     '?', '!', '(', '[', '</h1', '</h2', '</h3', '</article'
  # ]
  items = "(,[,file, ,sponsored,\t,img,inputs,promoted,sd,advertising,google,google_analytics,twitter,facebook,linkedin,pinterest,instagram,stumble_x,ads,native,buzzword,paid,adblock,youtube,adword,click,money,market,cpc,discover,social,fund,campaign,analytic,pay,track,traffic,budget,device,target,interest,age,gender,geographic,state,city,country,conversion,contact,ctr,precise,stumble_y,download,content,username,blogger,wordpress,feedburner,unpaid,html,css,php,asp,js,dwt,angular,query,com,edu,org,net,xml,json,jpg,png,svg,comment,reply,microsoft,apple,disqus,promotions,mobile,$,price,account,share,save,%,discount,deal,coupon,reddit,gif,omniture,video"
  items = items.split(',')
  prb.probe_html_items(items)
  

