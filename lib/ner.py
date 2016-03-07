#-*- coding:utf-8 -*-

# basic
import os, re, json, sys

# Language process
import nltk, ftfy
from nltk.tag.stanford import NERTagger
from nltk.tokenize import sent_tokenize
from nltk.corpus import stopwords
from nltk.stem.porter import *
stemmer = PorterStemmer()

from bs4 import BeautifulSoup

# [Stanford CoreNLP]
snlp_path = '/home/marsan/workspace/stanford_nlp'
sys.path.append("%s/stanford_corenlp_pywrapper/stanford_corenlp_pywrapper" % snlp_path)
from stanford_corenlp_pywrapper import CoreNLP

#==========================================
#   main class
#==========================================  
class ner(object):
  def __init__(self, lang='en', en_ner=False):
    # feature parameters
    self.lang = lang

    # [NLTK wrapper for Stanford CoreNLP] (too slow, results soso.)
    if en_ner == 'nltk':
      self.entity_cols = ['PERSON', 'ORG', 'LOCATION', 'FACILITY', 'GPE']
      self.sner_root = '/home/marsan/workspace/stanford_nlp/stanford-ner-2015-04-20'
      self.sner_classifier = self.sner_root+'/classifiers/english.all.3class.distsim.crf.ser.gz'
      self.sner_main = self.sner_root+'/stanford-ner.jar'
      self.st = NERTagger(self.sner_classifier, self.sner_main, encoding='utf-8')

    # [Stanford CoreNLP pywrapper] (still slow, reaults too noisy)
    if en_ner == 'corenlp':
      self.entity_cols = ['LOCATION', 'TIME', 'PERSON', 'ORGANIZATION', 'MONEY', 'PERCENT', 'DATE']
      self.snlp = CoreNLP("ner", corenlp_jars=["%s/stanford-corenlp-full-2015-04-20/*" % snlp_path])


  #===========================================
  #   Standford CoreNLP pywrapper
  #===========================================
  def get_ner_stanford_corenlp(self, txt):
    tree = self.snlp.parse_doc(txt.upper())
    ners = {n: [] for n in self.entity_cols}
    results = [list(zip(r['ner'], r['tokens'])) for r in tree['sentences']]
    results = [(k[0], k[1].lower()) for v in results for k in v if k[0] in self.entity_cols]
    ners = {k: [] for k in self.entity_cols}
    for k,v in results: ners[k].append(v)
    ners = {k: list(set(v)) for k,v in ners.items()}
    return ners

  # #===========================================
  # #   Standford CoreNLP (slow but better)
  # #===========================================
  def get_ner_tags(self, text):
    ners = {}
    terms = [(k,v) for k,v in self.st.tag(text.split()) if v != 'O']
    for t in self.entity_cols:
      ners[t] = list(set([re.sub('[^0-9a-zA-Z]+', ' ', k.lower()) for k,v in terms if v == t]))
    return ners

  #===========================================
  #   NLTK NER (very bad accuracy, a lot garbage)
  #===========================================
  def get_ner_nltk(self, text):
    sents = nltk.sent_tokenize(text)  # sentences
    tokenized_sents = [nltk.word_tokenize(s) for s in sents]
    tagged_sents = [nltk.pos_tag(s) for s in tokenized_sents]
    chunked_sents = [x for x in nltk.ne_chunk_sents(tagged_sents)]
    raw = self.traverseTree(chunked_sents)
    ners = {}
    for n in self.entity_cols: ners[n] = []
    [ners[k].append(v.lower()) for k,v in raw]
    for n in self.entity_cols: ners[n] = list(set(ners[n]))
    return ners

  def traverseTree(self, tree):
    result = []
    for subtree in tree:
      if type(subtree) == nltk.tree.Tree:
        if subtree.label() in self.entity_cols:
          result += [(subtree.label(), subtree[0][0])]
        else:
          result += (self.traverseTree(subtree))
    return result
