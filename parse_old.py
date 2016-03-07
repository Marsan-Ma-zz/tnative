#-*- coding:utf-8 -*-
import os, re, sys, zipfile, random, timeit, pickle, json, urllib.parse, time
import ftfy
import dateutil.parser
import multiprocessing as mp
import numpy as np

from math import exp, log, floor
from bs4 import BeautifulSoup as bs
from datetime import datetime, timedelta
from collections import Counter, OrderedDict
from itertools import islice
from newspaper import Article
from gensim.models import word2vec, doc2vec, Doc2Vec

pool_size = mp.cpu_count()
mylib_path = '/home/marsan/workspace/tnative'
sys.path.append(mylib_path)

from lib import schema as db
from lib import dgen as dgen
from lib import ner as ner
from lib import scraper as scraper
from lib import nlp as nlp

class parse_machine(object):
  def __init__(self, path='', idx=0):
    self.idx = idx
    self.filepath = path

  #-------------------------
  #   parse raw
  #-------------------------
  def fix_html(self, html):
    html = ftfy.fix_text(html)
    if (html.count("   ")*3 > len(html)): html.replace('   ', '')   # bad html special case
    html = html.replace('\\r', ' ').replace('\\n', ' ').replace('\\t', ' ')   # remove spaces
    html = re.sub(' +', ' ', html)
    return html

  def gen_html(self, pids=None):
    files = zipfile.ZipFile(self.filepath)
    idx = 0
    for fname in files.namelist():
      # check filename
      if (len(fname) < 3): continue
      fid = int(fname.split('_')[0].split('/')[1])
      if pids:
        if fid not in pids: continue
      # check html
      html = str(files.open(fname, 'r').read())
      # check size for mongodb document 16mb limit
      if (len(html) > 2**22): 
        bigfilename = "./toobig/%s" % int(fname.split('_')[0].split('/')[1])
        with open(bigfilename, 'wt') as f:
          f.write(html)
        print("write toobig file in %s" % bigfilename)
        html = html[:2**22]   # half of limit
      yield idx, fname, html
      idx += 1

  def tag_link(self, link):
    if not link: return ''
    link = re.sub('[^0-9a-zA-Z_-]+', ' ', link.lower())
    if '//' in link: link = link.split('//')[1]
    if '?' in link: link = link.split('?')[0]
    link = '/'.join(link.split('/')[0:1])
    return link
  
  def parse_links(self, html):
    # html = ' '.join(html.split())  # remove spaces & newlines
    link_segs = [re.sub('[^0-9a-zA-Z]+', ' ', self.tag_link(match)).split() for match in re.findall('http[s]?://[^\s<>"]+|www.[^\s<>"]+',html)]
    link_segs = [j for k in link_segs for j in k if not j.isdigit()]
    link_stats = {k:v for k,v in Counter(link_segs).items() if ((len(k) > 3) & (v > 2))}
    return link_stats

  def process_date(self, date):
    date = date or ''
    if isinstance(date, str):
      try:
        if date.isdigit():
          if (len(date) > 10): date = date / 1e3
          date = datetime.fromtimestamp(int(date))
        elif date == '':
          date = datetime.fromtimestamp(0)
        else:
          # print '[dateutil.parser]', date
          date = dateutil.parser.parse(date[:19])
      except:
        date = None # giveup, hope other solution work.
    elif isinstance(date, int):
      if (len(str(date)) > 10): date = date / 1e3
      date = datetime.fromtimestamp(date)
    return date

  def parsein_raw(self, pids=None):
    items = []
    processes = []
    save_processes = []
    mp_pool = mp.Pool(pool_size)
    for idx, fname, html in self.gen_html(pids):
      p = mp_pool.apply_async(parser_nlp, (fname, html))
      processes.append(p)
      if (idx % 5000 == 0): 
        for p in processes:
          items.append(p.get())
        sp = mp_pool.apply_async(batch_create, (items, db.articles)) #, False))
        save_processes.append(sp)
        # items = self.batch_create(items, tbl=db.articles, remove_existed=False)
        processes = []
        items = []
        print("[parsein_raw] done %i @ %s" % (idx, datetime.now()))
    # save rest items
    for p in processes:
      items.append(p.get())
    if items: items = batch_create(items, tbl=db.articles) #, remove_existed=False)
    for sp in save_processes: 
      sp.get()  # wait all save done
    print("[parsein_raw] done all @ %s" % (datetime.now()))
  

  # def parsein_extra(self):
  #   for idx, fname, html in self.gen_html():
  #     fid = int(fname.split('_')[0].split('/')[1])
  #     a = db.articles.objects(fid=fid).first()
  #     if not a:
  #       print("[ERROR] fid=%i not exist ???" % fid)
  #     else:
  #       a.update(
  #         lines       = html.count('\n'),
  #         spaces      = html.count(' '),
  #         tabs        = html.count('\t'),
  #         braces      = html.count('{'),
  #         brackets    = html.count('['),
  #         quesmarks   = html.count('?'),
  #         exclamarks  = html.count('!'),
  #         words       = len(re.split('\s+', html)),
  #       )
  #     if (idx % 1000 == 0): print("parsein_extra %i done @ %s" % (idx, datetime.now()))
  #   print("[parsein_extra] done @ %s" % (datetime.now()))
  
  #-------------------------
  #   parse label
  #-------------------------
  def fname2id(self, fname):
    return int(fname.split('_')[0])

  def parse_isad(self):
    mylib_path = '/home/marsan/workspace/tnative'
    fname = '%s/data/train.csv' % mylib_path      
    with open(fname, 'r') as f:
      lines = f.readlines()[1:]
      isad_ids = [self.fname2id(fid) for fid, isad in [line.strip().split(',') for line in lines] if int(isad) > 0]
      notad_ids = [self.fname2id(fid) for fid, isad in [line.strip().split(',') for line in lines] if int(isad) == 0]

    print("%i isad_ids, %i notad_ids" % (len(isad_ids), len(notad_ids)))
    print("-----")
    print("isad_ids:", isad_ids[:10], "...")
    print("-----")
    print("notad_ids:", notad_ids[:10], "...")

    db.articles.objects(fid__in=isad_ids).update(set__isad=True)
    db.articles.objects(fid__in=notad_ids).update(set__isad=False)

    ad_cnt = db.articles.objects(isad=True).count()
    notad_cnt = db.articles.objects(isad=False).count()
    unknown_cnt = db.articles.objects(isad=None).count()
    print("marked docs: %i as ad, %i as notad, %i as unknown." % (ad_cnt, notad_cnt, unknown_cnt))

  #-------------------------
  #   mark empty & bad htmls according to forum intel: (https://www.kaggle.com/c/dato-native/forums/t/15832/what-do-we-do-with-empty-files)
  #-------------------------
  def mark_empty_bad_docs(self):
    # [empty html, 458 totally]
    mylib_path = '/home/marsan/workspace/tnative'
    with open('%s/data/0_to_4_empty_list.txt' % mylib_path, 'r') as f:
      lines = f.readlines()
      empty_ids = [int(l.split('_')[0]) for l in lines]
    raw = db.articles.objects(fid__in=empty_ids).no_cache()
    raw.update(status='empty')
    print("%i empty articles, %i updated as empty." % (len(empty_ids), raw.count()), empty_ids[:10])
    # [bad html, 12015 totally]
    mylib_path = '/home/marsan/workspace/tnative'
    with open('%s/data/bad_html.csv' % mylib_path, 'r') as f:
      lines = f.readlines()[1:]
      bad_ids = [int(l.replace('\"', '').split('_')[0]) for l in lines]
    raw = db.articles.objects(fid__in=bad_ids).no_cache()
    raw.update(status='bad')
    print("%i bad articles, %i updated as empty." % (len(bad_ids), raw.count()), bad_ids[:10])
    # [summary/check]
    print(db.articles.objects.no_cache().item_frequencies('status', normalize=False))


  # def fix_bad_docs(self):
  #   raw = db.articles.objects(status='bad').no_cache()
  #   for idx, r in enumerate(raw):
  #     html = r.html.replace('   ', '')
  #     html = bs(html).body.text
  #     html = urllib.parse.unquote(html)
  #     r.update(html=html)
  #     if (idx % 1000 == 0):
  #       print("[fix_bad_docs] %i done @ %s" % (idx, datetime.now()))



def batch_create(items, tbl, remove_existed=True):
  Ts = timeit.default_timer()
  if remove_existed:
    cids = [item.fid for item in items]
    existed_ids = [r.fid for r in tbl.objects(fid__in=cids).only('fid')]
    items = [item for item in items if (item.fid not in existed_ids)]
  else:
    existed_ids = []
  items_cnt = len(items)
  if (items_cnt > 0):
    tbl.objects.insert(items)
    print("[batch_create] create %i / skip %i existed items, cost %.3f secs @ %s" % (items_cnt, len(existed_ids), (timeit.default_timer() - Ts), datetime.now()))
  else:
    print("[batch_create] skip all %i existed items." % (len(existed_ids)))
  return []

#-------------------------
#   Advance features
#-------------------------
# def parser_advance(submit=False):
#   Ts = datetime.now()
#   idx = 0
#   docs = []
#   mp_pool = mp.Pool(pool_size)
#   # raw = db.articles.objects(isad=None) if submit else db.articles.objects(isad__ne=None)
#   # raw = raw.filter(processed__ne=True).no_cache().timeout(False)
#   raw = db.articles.objects.no_cache().timeout(False)
#   print("%i to go~ @ %s" % (raw.count(), datetime.now()))
#   for idx, r in enumerate(raw):
#     ### [single thread for debug]
#     # parser_nlp(r)
#     ### [multithread by pool]
#     docs.append(r)
#     idx += 1
#     if (idx % 100 == 0): 
#       it = mp_pool.imap(parser_nlp, docs)
#       ttl = 120
#       for i in range(len(docs)):
#         try:
#           result = it.next(timeout=ttl)
#           if result: print(result)
#         except Exception as e:
#           print("doc %s timeout: %s" % (docs[i].id, e))
#           ttl = 0.1
#       docs = []
#       print("[parser_advance] %i done, cost %s" % (idx, (datetime.now() - Ts)))


def sanitize_txt(txt, lower=False):
  txt = str(txt)
  if lower: txt = txt.lower()
  txt = txt.replace('\\r', ' ').replace('\\n', ' ').replace('\\t', ' ')
  txt = re.sub('[^0-9a-zA-Z\u4e00-\u9fa5]+', ' ', txt) # filter not numerical/alphabatic/chinese
  txt = re.sub(' +', ' ', txt)
  txt = re.sub('\+', ' ', txt)
  return txt


def parser_nlp(fname, html):
  Ts = timeit.default_timer()
  raw_html = html
  # basic info
  fid = int(fname.split('_')[0].split('/')[1])
  pm = parse_machine()
  html = pm.fix_html(html)
  link_stats = pm.parse_links(html)
  link_factors = [t for t in list(set(" ".join(link_stats.keys()).lower().split())) if (len(t) > 3)]
  doc = db.articles(
    fid           = fid,
    html          = html,
    html_cnt      = len(html),
    link_stats    = link_stats,
    link_factors  = link_factors,
    rand          = random.random(),
    # extra
    lines         = raw_html.count('\n'),
    spaces        = raw_html.count(' '),
    tabs          = raw_html.count('\t'),
    braces        = raw_html.count('{'),
    brackets      = raw_html.count('['),
    quesmarks     = raw_html.count('?'),
    exclamarks    = raw_html.count('!'),
    words         = len(re.split('\s+', raw_html)),
  )
  # check empty
  if ((doc.html == None) | (len(doc.html.replace(r'\s', '')) < 10)):
    doc.empty = True
    return doc
  try:
  # if True:
    pd = Article('', fetch_images=False)
    pd.set_html(doc.html)
    pd.parse()
    pd.nlp()
  except Exception as e:
    print("-"*60)
    print("[parser_nlp %s]: %s" % (doc.fid, e)) 
    print(doc.html[:500])
    print("-"*60) 
    return doc #"%s: %s" % (e, doc.id)
  # select cleaned_text
  cleaned_text = " ".join(pd.text.lower().split())
  if (len(cleaned_text) < 140):
    soup = bs(doc.html)
    if soup.body: 
      cleaned_text = soup.body.text
    if (len(cleaned_text) < 140): 
      cleaned_text = soup.text
  cleaned_text = sanitize_txt(cleaned_text, lower=True)
  bow = nlp.nlp().txt2words(cleaned_text or '', False)
  # save results 
  try:
    opengraph = pd.meta_data.get('og', {}) if pd.meta_data else {}
    top_image = opengraph.get('image') or (pd.top_image if pd.top_image else None)
    if isinstance(top_image, dict): top_image = top_image.get('identifier')
    if isinstance(opengraph.get('locale'), dict): opengraph['locale'] = opengraph.get('locale').get('identifier')
    publish_date = pm.process_date(opengraph.get('updated_time') or pd.publish_date)
    # canonical_link & domain
    domain = canonical_link = str(opengraph.get('url') or pd.canonical_link)
    if '//' in domain: domain = domain.split('//')[1]
    if '?' in domain: domain = domain.split('?')[0]
    domain = '/'.join(domain.split('/')[0:1])
    # update
    # doc.update(
    doc = db.articles(
      fid               = doc.fid,
      html              = doc.html,
      link_stats        = doc.link_stats,
      link_factors      = doc.link_factors,
      rand              = doc.rand,
      html_cnt          = doc.html_cnt,
      #
      lines             = doc.lines,
      spaces            = doc.spaces,
      tabs              = doc.tabs,
      braces            = doc.braces,
      brackets          = doc.brackets,
      quesmarks         = doc.quesmarks,
      exclamarks        = doc.exclamarks,
      words             = doc.words,
      #
      title             = str(opengraph.get('title') or pd.title)[:500],
      # cleaned_text      = str(cleaned_text),
      bow               = bow,
      tags              = [t.lower() for t in pd.tags],
      # opengraph         = {sanitize_txt(k): sanitize_txt(v) for k,v in opengraph.items()},
      # summary           = str(pd.summary),
      keywords          = pd.keywords,
      top_image         = str(top_image),
      movies            = pd.movies,
      publish_date      = publish_date,
      meta_site_name    = str(opengraph.get('site_name')),
      meta_lang         = str(opengraph.get('locale') or pd.meta_lang),
      meta_description  = str(opengraph.get('description') or pd.meta_description),
      meta_keywords     = pd.meta_keywords,
      canonical_link    = canonical_link,
      domain            = domain,
      authors           = [n.lower().replace(' ', '_') for n in pd.authors],
    )
  except Exception as e:
    print("-"*60)
    print("[Error] while [%s] in parser_nlp: %s" % (doc.id, e))
    data = {
      "title"     : str(opengraph.get('title') or pd.title)[:500],
      "text"      : cleaned_text[:140],
      "tags"      : [t.lower() for t in pd.tags],
      "opengraph" : opengraph,
      "summary"   : str(pd.summary),
      "keywords"  : pd.keywords,
      "top_image" : str(top_image),
      "movies"    : pd.movies,
      "date"      : publish_date, #opengraph.get('updated_time') or pd.publish_date,
      "site_name" : str(opengraph.get('site_name')),
      "locale"    : str(opengraph.get('locale') or pd.meta_lang),
      "desc"      : str(opengraph.get('description') or pd.meta_description),
      "keywords"  : pd.meta_keywords,
      "url"       : canonical_link,
      "authors"   : pd.authors,
    }
    for k,v in data.items():
      print(k, v, v.__class__)
    print("-"*60)
  return doc
    
#-----------------------------
#   Named entity recognition
#-----------------------------
def doc_ner_star(pstar):
  doc_ner(*pstar)

def doc_ner(rmin, rmax, submit):
  Ts = datetime.now()
  raw = db.articles.objects(isad__ne=None) if submit else db.articles.objects(isad=None)
  raw = raw.no_cache().filter(rand__gte=rmin, rand__lt=rmax, processed__ne=True)
  tgr = ner.ner()
  for idx, b in enumerate(raw.timeout(False)):
    if not b.html.replace(r'\s', ''): 
      b.update(processed=True, empty=True)
      continue
    b.cleaned_text = str(b.cleaned_text)  # turn None in 'None'
    if (len(b.cleaned_text) > 300):
      text = "%s. %s" % (b.title, b.cleaned_text)
    else:
      soup = bs(b.html, 'lxml')
      soup = soup.body or soup
      text = "%s. %s" % (b.title, soup.text)
    text = ' '.join(text.split())   # remove spaces and newlines
    try:
      entries = tgr.get_ner_stanford_corenlp(text) 
      print(b.id, entries)
      b.update(
        # ner_facility      = entries['FACILITY'],
        # ner_gpe           = entries['GPE'],
        ner_location      = entries['LOCATION'],
        ner_org           = entries['ORGANIZATION'],
        ner_person        = entries['PERSON'],
        # ner_date          = entries['DATE'],
        processed         = True,
      )
    except Exception as e:
      print("[Error] doc_id:%s in doc_ner: %s" % (b.id, e))
    if (idx % 100 == 0): 
      print("[doc_ner] %i done, cost %s" % (idx, (datetime.now() - Ts)))
    

def get_ners(tbl=db.articles, submit=False, psize=4):
  mp_pool = mp.Pool(psize)
  params = []
  pre_i = 0
  for i in np.linspace(0, 1, 11)[1:]:
    params.append([pre_i, i, submit])
    pre_i = i
  print(params)
  mp_pool.map(doc_ner_star, params)
  
#-------------------------
#   find rare labels
#-------------------------
def find_rare_labels(th=10):
  tgr = dgen.data_gen(False)
  for atr in ['title', 'tags', 'summary', 'keywords', 'top_image', 'movies', 
              'meta_site_name', 'meta_lang', 'meta_description', 'meta_keywords', 'canonical_link', 'authors']:
    print("=====[find rare %s] @ %s=====" % (atr, datetime.now()))  
    Ts = datetime.now()
    bobow = [tgr.tag_text(getattr(r, atr)) for r in db.articles.objects.no_cache().only(atr).timeout(False)]
    wordcount = Counter([item for sublist in bobow for item in sublist])
    rare_words = [k for k,v in wordcount.items() if v <= th]
    rare_cnt, all_cnt = len(rare_words), len(wordcount)
    print("rare/all=%i/%i, cost %s" % (rare_cnt, all_cnt, datetime.now() - Ts), rare_words[:30])
    rare_fname = './rares/%s.pkl' % (atr)
    pickle.dump(rare_words, open(rare_fname, 'wb'))


def prepare_fh_fast(mode_name, D=2**24):
  dgen.gen_fast_data(mode_name, D, debug=False)


#-------------------------
#   stat author behavior
#   (ensured: no need to lower, also no need to replace space between name. names are all unique)
#-------------------------
def stat_authors():
  Ts = datetime.now()
  # gather raw
  ad_authors = [] 
  nonad_authors = []
  stat = {'ad_has_author': 0, 'ad_no_author': 0, 'nonad_has_author': 0, 'nonad_no_author': 0}
  raw = db.articles.objects(isad__ne=None, rand__lt=0.8) #if submit else db.articles.objects(isad__ne=None)
  for r in raw.no_cache().only('isad', 'authors').timeout(False):
    # authors = [a.lower() for a in list(set(r.authors))]   # <-- DON'T USE, cause cannot find them in db
    authors = list(set(r.authors))
    has_author = (len(authors) > 0)
    # author
    if r.isad: 
        ad_authors += authors
    else:
        nonad_authors += authors
    # stat
    if (r.isad & has_author):
        stat['ad_has_author'] += 1
    elif (r.isad & (not has_author)):
        stat['ad_no_author'] += 1
    elif ((not r.isad) & has_author):
        stat['nonad_has_author'] += 1
    elif ((not r.isad) & (not has_author)):
        stat['nonad_no_author'] += 1
  print(stat)
  # stat       
  ad_authors_stat = Counter(ad_authors) 
  nonad_authors_stat = Counter(nonad_authors) 
  all_author_names = list(set(list(ad_authors_stat.keys()) + list(nonad_authors_stat.keys())))
  all_authors_stat = {k: {'ad': ad_authors_stat[k], 'nonad': nonad_authors_stat[k]} for k in all_author_names}
  # summarize & pickle
  author_memo = {
    'pure_ad_authors':      {k: v for k,v in all_authors_stat.items() if ((v['nonad'] == 0) & (v['ad'] > 5))},
    'pure_nonad_authors':   {k: v for k,v in all_authors_stat.items() if ((v['ad'] == 0) & (v['nonad'] > 5))},
    'prefer_ad_authors':    {k: v for k,v in all_authors_stat.items() if ((v['nonad'] > 0) & (v['ad']/2 > v['nonad']))},
    'prefer_nonad_authors': {k: v for k,v in all_authors_stat.items() if ((v['ad'] > 0) & (v['nonad']/2 > v['ad']))},
  }
  for k,v in author_memo.items():
    anames = v.keys()
    items = db.articles.objects(authors__in=anames)
    print("%s has %i to update." % (k, items.count()))
    exec("items.update(is_%s=True)" % k) in locals()

#-------------------------
#   stat domain behavior
#-------------------------
def stat_domains():
  def tag_link(link):
    if not link: return ''
    if '//' in link: link = link.split('//')[1]
    if '?' in link: link = link.split('?')[0]
    link = '/'.join(link.split('/')[0:1])
    return link

  raw = db.articles.objects(isad__ne=None, rand__lt=0.8).no_cache() # if submit else db.articles.objects(isad__ne=None)
  raw = raw.filter(canonical_link__nin=[None, '']).only('canonical_link', 'isad').timeout(False)
  canonical_links = [[r.canonical_link, r.isad] for r in raw]
  print(len(canonical_links))

  ad_domain = [tag_link(url) for url, label in canonical_links if label]
  nad_domain = [tag_link(url) for url, label in canonical_links if not label]
  ad_domain_stat = {k: v for k,v in Counter(ad_domain).items() if ((k != '') & (v >= 3))}
  nad_domain_stat = {k: v for k,v in Counter(nad_domain).items() if ((k != '') & (v >= 3))}

  mix_domains = {k: [v, nad_domain_stat[k]] for k,v in ad_domain_stat.items() if k in nad_domain_stat}
  domain_memo = {
    'prefer_nonad_domains'  : {k: v for k,v in mix_domains.items() if (v[1]/2 > v[0])},
    'prefer_ad_domains'     : {k: v for k,v in mix_domains.items() if (v[0]/2 > v[1])},
    'pure_ad_domains'       : {k: v for k,v in ad_domain_stat.items() if ((k not in nad_domain) & (v >= 3))},
    'pure_nonad_domains'    : {k: v for k,v in nad_domain_stat.items() if ((k not in ad_domain) & (v >= 3))},
  }
  for k,v in domain_memo.items():
    print(k, len(v))
    anames = v.keys()
    items = db.articles.objects(domain__in=anames)
    print("%s has %i to update." % (k, items.count()))
    exec("items.update(set__is_%s=True)" % k) in locals()

#-------------------------
#   stat links behavior
#-------------------------
def count_link_elements(raw):
  links = [l for r in raw for l in r.link_factors]
  links_stats = {k: v for k,v in Counter(links).items() if ((v > 10) & (len(k) > 3))}
  return links_stats


def stat_link_factors():
  # [raw counts]
  Ts = datetime.now()
  ad_links_stats = count_link_elements(db.articles.objects(isad=True, rand__lt=0.8).no_cache().only('link_factors'))
  print('ad_links_stats', len(ad_links_stats), datetime.now() - Ts)
  Ts = datetime.now()
  nad_links_stats = count_link_elements(db.articles.objects(isad=False, rand__lt=0.8).no_cache().only('link_factors'))
  print('nad_links_stats', len(nad_links_stats), datetime.now() - Ts)
  # [categorize]
  th = 100
  ad_links = set([k for k,v in ad_links_stats.items() if v > th])
  nad_links = set([k for k,v in nad_links_stats.items() if v > th])
  intersection = list(ad_links.intersection(nad_links))
  print('ad_links:%i/nad_links:%i/mixed:%i' % (len(ad_links), len(nad_links), len(intersection)))
  results = {
    'pure_ad_links': list(ad_links - nad_links),
    'pure_nad_links': list(nad_links - ad_links),
    'prefer_ad_links': [l for l in intersection if (ad_links_stats[l]/2 > nad_links_stats[l])],
    'prefer_nad_links': [l for l in intersection if (nad_links_stats[l]/2 > ad_links_stats[l])],
  }
  # [save to db]
  for idx, r in enumerate(db.articles.objects.no_cache()):
    r.update(
      has_pure_ad_links         = (r.link_factors in results['pure_ad_links']),
      has_prefer_ad_links       = (r.link_factors in results['prefer_ad_links']),
      has_pure_nad_links        = (r.link_factors in results['pure_nad_links']),
      has_prefer_nad_links      = (r.link_factors in results['prefer_nad_links']),
    )
    if (idx % 10000 == 0): print('[stat_link_factors]', idx, datetime.now())


#-------------------------
#   stat_lang_ad_freq
#-------------------------
def stat_lang_ad_freq():
  # collect raw counts
  def count_lang(raw):
    stat = {} 
    for k,v in raw.item_frequencies('meta_lang').items():
      k = str(k)[:2].lower()
      stat[k] = stat[k] + v if k in stat else v
    return stat
  ad_langs = count_lang(db.articles.objects(isad=True, rand__lt=0.8).no_cache())
  all_langs = count_lang(db.articles.objects(isad__ne=None, rand__lt=0.8).no_cache())
  
  # group bins & label
  def log_divide(k, v):
    x = float(k)/v if k else 10
    x = floor(log(abs(x))**2)
    return x
  langs_stat = {k: (ad_langs.get(k, 0), v) for k,v in all_langs.items()}
  langs_stat = {k: log_divide(v[0], v[1]) for k,v in langs_stat.items()}
  langs_cat = {j: [k for k,v in langs_stat.items() if v == j] for j in langs_stat.values()}
  print(langs_cat)
  
  # update
  for idx, r in enumerate(db.articles.objects.no_cache()):
    lang = str(r.meta_lang)[:2].lower()
    r.update(lang_adrate_group=str(langs_stat.get(lang)))
    # print("update %s as %s" % (lang, langs_stat[lang]))
    if (idx % 10000 == 0): print('[stat_lang_ad_freq]', idx, datetime.now())
  return langs_cat


#-------------------------
#   social counts
#-------------------------
def parse_social_cnts():  
  docs = db.articles.objects(canonical_link__nin=[None, ''], processed__ne=True).no_cache().timeout(False) #.limit(3000)
  # print("[parse_social_cnts] %i to go~ @ %s" % (docs.count(), datetime.now()))
  urls = []
  docs_pool = {}
  scrp = scraper.scraper()
  cnt = {'success': 0, 'fail': 0, 'skip': 0, 'total': len(urls)}
  for idx, r in enumerate(docs):
    if not scrp.validate_url(r.canonical_link): 
      cnt['skip'] += 1
      continue
    url = urllib.parse.quote(r.canonical_link.replace("\'", ''))
    urls.append(url)
    docs_pool[url] = r
    if len(urls) >= 1000:
      scores = scrp.batch_get_fb_info(urls)
      # print(scores)
      if not scores:  # retry
        time.sleep(60)
        scores = scrp.batch_get_fb_info(urls)
      for s in scores:
        u = urllib.parse.quote(s['url'])
        doc = docs_pool.get(u)
        if not doc: 
          print("[Error] can't find doc url="+u)
          cnt['fail'] += 1
          continue
        else:
          doc.update(
            fb_click_count            = s['click_count'],
            fb_comment_count          = s['comment_count'],
            fb_like_count             = s['like_count'],
            fb_share_count            = s['share_count'],
            fb_total_count            = s['total_count'],
            processed                 = True,
          )
          cnt['success'] += 1
      print(idx, cnt, datetime.now())
      urls = []
      docs_pool = {}


#-------------------------
#   alexa rank
#-------------------------
def parse_alexa_rank():  
  docs = db.articles.objects(canonical_link__nin=[None, ''], processed__ne=True).no_cache().timeout(False) #.limit(3000)
  urls = []
  docs_pool = {}
  scrp = scraper.scraper()
  cnt = {'success': 0, 'fail': 0, 'skip': 0, 'total': len(urls)}
  for idx, r in enumerate(docs):
    if not scrp.validate_url(r.canonical_link): 
      cnt['skip'] += 1
      continue
    # url = urllib.parse.quote(r.canonical_link.replace("\'", ''))
    url = r.canonical_link
    urls.append(url)
    docs_pool[url] = r
    if len(urls) >= 1000:
      scores = scrp.batch_get_alexa_rank(urls)
      # print(scores)
      for s in scores:
        u = s['url'] #urllib.parse.quote(s['url'])
        doc = docs_pool.get(u)
        if not doc: 
          print("[Error] can't find doc url="+u)
          cnt['fail'] += 1
          continue
        else:
          doc.update(
            alexa_rank      = s['rank'],
            alexa_delta     = s['delta'],
            alexa_loc_name  = s['loc_name'],
            alexa_loc_rank  = s['loc_rank'],
            processed       = True,
          )
          cnt['success'] += 1
      print(idx, cnt, datetime.now())
      urls = []
      docs_pool = {}


#-------------------------
#   word2vec
#-------------------------
def thread_update(rec, key, val):
  exec("rec.update(%s=val)" % key)

def train_doc2vec(test_cnt=0):
  print("[train_doc2vec] start @ %s" % datetime.now())
  nl = nlp.nlp()
  raw = db.articles.objects().timeout(False).no_cache()
  if test_cnt: raw = raw.limit(test_cnt)
  docs = (doc2vec.LabeledSentence(words=(r.bow or []), tags=[str(r.id)]) for r in raw.only('bow'))
  model = nl.train_doc2vec(docs, save=True)
  trained_ids = list(model.docvecs.doctags.keys())
  mp_pool = mp.Pool(pool_size*8)
  processes = []
  for idx, r in enumerate(raw):
    rid = str(r.id)
    if rid not in trained_ids: continue
    rvec = [float(v) for v in model.docvecs[rid]]
    # r.update(doc2vec=rvec)
    p = mp_pool.apply_async(thread_update, (r, 'doc2vec', rvec))
    processes.append(p)
    if (idx % 1000 == 0): 
      for p in processes: r = p.get()
      print("[train_doc2vec] update %i done @ %s" % (idx, datetime.now()))
  print("[train_doc2vec] completed @ %s" % datetime.now())
  return model

def train_tfidf(test_cnt=0):
  nl = nlp.nlp()
  raw = db.articles.objects().timeout(False).no_cache()
  if test_cnt: raw = raw.limit(test_cnt)
  docs = (" ".join(r.bow or []) for r in raw.only('bow'))
  model = nl.train_tfidf(docs, save=True)
  mp_pool = mp.Pool(pool_size*8)
  processes = []
  for idx, r in enumerate(raw):
    tvec = model.transform([" ".join(r.bow or [])])
    tvec = list(tvec.getrow(0).toarray()[0])
    # r.update(tfidf_vec=tvec)
    p = mp_pool.apply_async(thread_update, (r, 'tfidf_vec', tvec))
    processes.append(p)
    if (idx % 1000 == 0):
      for p in processes: r = p.get()
      print("[train_tfidf] update %i done @ %s" % (idx, datetime.now()))
  print("[train_tfidf] all completed @ %s" % datetime.now())
  return model


#==========================================
#   test main
#==========================================
def parse_zipfile(i):
  c = parse_machine('./data/%i.zip' % (i), i)

  # missing ids
  with open(mylib_path+'/data/missing.ids', 'rt') as f:
    lines = f.readlines()
    pids = [int(i) for i in lines]
  existed_pids = [r.fid for r in db.articles.objects(fid__in=pids).only('fid')]
  pids = [i for i in pids if i not in existed_pids]
  print("%i pids to go~" % len(pids))
  c.parsein_raw(pids=pids)
  # c.parsein_extra()


if __name__ == '__main__':  
  cmd = str(sys.argv[1]) if (len(sys.argv) > 1) else None

  # ### [parse raw html]
  ### db.articles.drop_collection()
  print("currently have %i articles, %i training samples, %i prepared samples @ %s" % (db.articles.objects.count(), db.articles.objects(isad__ne=None).count(), db.samples.objects.count(), datetime.now()))
  mp_pool = mp.Pool(pool_size)
  mp_pool.map(parse_zipfile, range(6))
  ### for i in range(6): parse_zipfile(i)
  pm = parse_machine()
  pm.parse_isad()
  pm.mark_empty_bad_docs()
  # ### pm.fix_bad_docs() # --use ftfy instead
  # ### parser_advance()  # --merged to parse_raw

  
  # # ### [data engineering features]  
  # db.articles.objects(processed=True).update(set__processed=False)
  parse_social_cnts()
  stat_authors() # very fast
  stat_domains() # very fast
  stat_lang_ad_freq()
  stat_link_factors() # very slow

  ### [topic modeling]
  ### html2bow()   # --merged to parse_raw
  # train_tfidf()
  # train_doc2vec()

  # ### [remove blacklist]
  # find_rare_labels()  # very memory consumer, and no performance
  # prepare_fh('D_20_beta_rm_blacklist_w_stats', D=2**24, remove_blacklist=True)

  # ### [prepare fast examples (hashed features)]
  # prepare_fh_fast('D_24_tfidf', D=2**24)
  # prepare_fh_fast('D_24_tfidf_doc2vec_cnts', D=2**24)


  #------------------------------------------------------------------------
  ### [not used yet, skip.]
  # db.articles.objects(processed=True).update(set__processed=False)
  # get_ners(submit=submit) # very, very slow ...
  # db.articles.objects(processed=True).update(set__processed=False)
  # parse_alexa_rank()
  
          
