#-*- coding:utf-8 -*-

import os, re, json, sys, pickle
import nltk, ftfy
import nltk.data
import multiprocessing as mp
pool_size = mp.cpu_count()

from itertools import tee
from datetime import datetime
from nltk.stem.porter import *
from nltk.corpus import stopwords
from bs4 import BeautifulSoup
from sklearn.feature_extraction.text import CountVectorizer
from gensim.models import word2vec, doc2vec, Doc2Vec
from sklearn.feature_extraction.text import TfidfVectorizer

root_path = '/home/marsan/workspace/tnative'
sys.path.append(root_path)

# check gensim has supported scipy version (for good ATLAS)
import gensim
assert gensim.models.doc2vec.FAST_VERSION > -1, "this will be painfully slow otherwise"

#==========================================
#   main class
#==========================================  
class nlp(object):
  def __init__(self):
    self.stemmer = PorterStemmer()
    self.langs = ['english', 'french', 'german', 'spanish']
    self.stopwords = set(j for k in [stopwords.words(lan) for lan in self.langs] for j in k)
    self.tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
    self.timestamp = datetime.now().strftime("%Y%m%d_%H%M")
  #-------------------------
  #   txt process
  #-------------------------
  def save_stem(self, word):
    try:
      return self.stemmer.stem(word)
    except Exception as e:
      print('nlp cannot stem: %s, because: %s' % (word, e))
      return word   # if stemmer error, keep origin word.


  def txt2words(self, txt, remove_stopwords=True):
    txt = BeautifulSoup(txt).get_text()
    txt = ftfy.fix_text(txt)
    txt = txt.replace("\\n", '')
    txt = re.sub("[^0-9a-zA-Z]"," ", txt)
    if remove_stopwords:
      words = [self.save_stem(w) for w in txt.lower().split() if (w not in self.stopwords) & (len(w) > 2) & (not w.isdigit())]
    else:
      words = [self.save_stem(w) for w in txt.lower().split() if (len(w) > 2) & (not w.isdigit())]
    return words


  def doc2sents(self, doc, remove_stopwords=False): # if for word2vec, DON'T remove_stopwords !!
    raw_sents = self.tokenizer.tokenize(doc.strip())
    sentences = []
    for sent in raw_sents:
      if len(sent) == 0: continue
      sentences.append(self.txt2words(sent, remove_stopwords=remove_stopwords))
    return sentences


  #-------------------------
  #   extract docs features
  #-------------------------
  # docs is list of raw text, Ex: docs = [r.html for r in db.articles.objects(html__nin=['', None]).limit(100)]
  def docs2feats(self, docs, max_features=5000):
    print("[docs2feats] start @ %s" % datetime.now())
    doc_words = []
    for d in docs:
      words = self.txt2words(d)
      doc_words.append(" ".join(words))
    vectorizer = CountVectorizer(analyzer="word", max_features=max_features) 
    doc_features = vectorizer.fit_transform(doc_words)
    vocab = vectorizer.get_feature_names()
    print("model shape:", doc_features.shape, ", vocabs: ", vocab[:30], "...")
    print("[docs2feats] done @ %s" % datetime.now())
    return doc_features

  #-------------------------
  #   tfidf
  #-------------------------
  def train_tfidf(self, docs, max_df=0.95, min_df=2, max_features=1000, save=False):
    print("[train_tfidf] start @ %s" % datetime.now())
    vectorizer = TfidfVectorizer(max_df=max_df, min_df=min_df, max_features=max_features)
    vectorizer.fit(docs)
    print('vectorizer features: ', vectorizer.get_feature_names()[:10], '...')
    if save:
      tfile = root_path+'/models/tfidf_model.pkl'
      pickle.dump(vectorizer, open(tfile, 'wb'), protocol=4)  # protocal=4 for objects > 4GB
      print("tfidf model dumped: %s" % tfile)
    print("[train_tfidf] done @ %s" % datetime.now())
    return vectorizer

  #-------------------------
  #   word2vec
  #-------------------------
  # num_features:   Word vector dimensionality                      
  # min_word_count: Minimum word count                        
  # n_jobs:         Number of threads to run in parallel
  # context:        Context window size                                                                                    
  # downsampling:   Downsample setting for frequent words
  # replace:        'True' if NOT plan to train the model any further, will train faster.
  def train_word2vec(self, sentences, num_features=300, min_word_count=40, context=10, downsampling=1e-3, n_jobs=16, replace=True, save=False):
    print("[train_word2vec] start @ %s" % datetime.now())
    model = word2vec.Word2Vec(sentences, workers=n_jobs, size=num_features, min_count=min_word_count, window=context, sample=downsampling)
    model.init_sims(replace=replace)   
    if save:
      model_name = root_path + "/models/%ifeatures_%iminwords_%icontext_%s.word2vec" % (num_features, min_word_count, context, self.timestamp)
      model.save(model_name)    
    ### some test
    # print(model.doesnt_match("div document woman child kitchen".split()))
    # print(model.most_similar("buy"))
    # print(model["buy"])
    print(model.index2word[:15])
    print("[train_word2vec] done @ %s" % datetime.now())
    return model


  def load_word2vec_pretrained_model(self, model_name='googlenews'):
    print("[load_word2vec_pretrained_model] start @ %s" % datetime.now())
    model_path = "/home/marsan/workspace/gensim/"
    if model_name == 'googlenews':
      model_path += "GoogleNews-vectors-negative300.bin.gz"
    elif model_name == 'freebase':  # strange result...
      model_path += "freebase-vectors-skipgram1000-en.bin.gz"
    else:
      print("[Error] unknown model name: %s " % model_name)
    model = word2vec.Word2Vec.load_word2vec_format(model_path, binary=True)
    print("[load_word2vec_pretrained_model] done @ %s" % datetime.now())
    return model


  #-------------------------
  #   doc2vec
  #-------------------------
  def choose_doc2vec_model(self, model_name='PV-DBOW', size=50):
    if model_name == 'PV-DM-concat':
      # PV-DM w/concatenation - window=5 (both sides) approximates paper's 10-word total window size
      model = Doc2Vec(dm=1, dm_concat=1, size=size, window=5, negative=5, hs=0, min_count=2, workers=pool_size)
    elif model_name == 'PV-DBOW':
      # PV-DBOW 
      model = Doc2Vec(dm=0, size=size, negative=5, hs=0, min_count=2, workers=pool_size)
    elif model_name == 'PV-DM-avg':
      # PV-DM w/average
      model = Doc2Vec(dm=1, dm_mean=1, size=size, window=10, negative=5, hs=0, min_count=2, workers=pool_size)
    return model


  def train_doc2vec(self, docs, alpha=0.025, min_alpha=0.025, epoch=10, learn_rate_delta=0.002, save=False):
    print("[train_doc2vec] for docs, start @ %s" % (datetime.now()))
    docs, docs2 = tee(docs)
    model = self.choose_doc2vec_model(model_name='PV-DBOW')
    model.build_vocab(docs2)
    for epo in range(epoch):
      docs, docs2 = tee(docs)
      print("train_doc2vec epoch %i start @ %s" % (epo, datetime.now()))
      model.train(docs2)
      model.alpha -= learn_rate_delta  # decrease the learning rate
      model.min_alpha = model.alpha  # fix the learning rate, no decay
    # print(model.docvecs[0])
    if save:
      model_name = root_path + "/models/alpha_%i_epoch_%i_%s.doc2vec" % (int(alpha*1000), epoch, self.timestamp)
      model.save(model_name)
      pickle.dump(model, open(model_name+'.pkl', 'wb'), protocol=4)  # protocal=4 for objects > 4GB
      print("model saved in %s" % model_name)
    print("[train_doc2vec] done @ %s" % datetime.now())
    return model


