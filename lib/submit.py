import sys, gzip, json, os, math, pickle, re, copy, itertools
import numpy as np

from datetime import datetime

root_path = '/home/marsan/workspace/tnative'
sys.path.append(root_path)
from lib import schema as db
from lib import top as top
from lib import dgen as dgen
from lib import learner as mdl
from lib import grader as grader
#==========================================
#   submit process
#==========================================  
class submitter(object):
  def __init__(self, outFilename='', smin=0, smax=1, en_fast_data=False, en_fast_load=True, sampleFile='sampleSubmission.csv'):
    self.smin = smin
    self.smax = smax
    self.sampleFile = root_path+'/data/%s' % sampleFile
    self.submitFile = root_path+'/submit/'+outFilename+'_'+datetime.now().strftime("%Y_%m%d_%H%M")+'.csv'
    self.en_fast_data = en_fast_data
    self.en_fast_load = en_fast_load

  def verify_model(self, model_pkl, vmin=0.9, vmax=1.0, en_plot=False):
    # load dgen
    model_pkl = "%s/models/%s.pkl" % (root_path, model_pkl)
    alg = model_pkl.split('_')[0]
    learner, info = pickle.load(open(model_pkl, 'rb'))
    print(info)
    if self.en_fast_data:
      dg = dgen.data_gen(D=learner.D, en_fast_data=self.en_fast_data, submit=False)
    else:
      dg = dgen.data_gen(D=learner.D, en_fast_load=self.en_fast_load, submit=False)
    raw = dg.gen_data(vmin, vmax, hashing=alg)
    y2p = learner.train(raw, training=False)
    grader.grader(en_plot=en_plot).auc_curve(y2p)
    return y2p, learner, dg


  def predict_file(self, model_pkl, submit=True):
    # load model & get features
    print("load model @ %s ..." % datetime.now())
    alg = model_pkl.split('_')[0]
    model_pkl = root_path+'/models/%s' % model_pkl
    learner, info = pickle.load(open(model_pkl, 'rb'))

    # data generate
    print("start data generate @ %s ..." % datetime.now())
    if self.en_fast_data:
      dg = dgen.data_gen(D=learner.D, en_fast_data=self.en_fast_data, submit=submit)
    else:
      dg = dgen.data_gen(D=learner.D, en_fast_load=self.en_fast_load, submit=submit)
    print('hashing alg:', alg)
    raw = dg.gen_data(self.smin, self.smax, hashing=alg, extra=True)
    # ids = [str(r.fid) for r in dg.raw_range(self.smin, self.smax).only('fid')]
    raw, raw2 = itertools.tee(raw)
    ids = [str(extra['fid']) for idx, x, y, extra in raw2]
    print("id count: ", len(ids))

    # read submit sample
    print("read submit sample @ %s ..." % datetime.now())
    with open(self.sampleFile, 'rt') as f:
      lines = [line.split('_')[0] for line in f.readlines()[1:]]
    
    # predict
    y2p = learner.train(raw, training=False, info={'all_cnt': len(ids)})
    answers = {ids[idx]: p for idx, (y,p) in enumerate(y2p)}
    print("check answers cnts: ", len(ids), len(y2p), len(answers))
    # return answers, y2p, ids  # for debug

    # generate submit file
    print("generate submit file @ %s ..." % datetime.now())
    with open(self.submitFile, 'wt') as f:
      f.write('file,sponsored\n')
      for l in lines:
        f.write("%s_raw_html.txt,%.15f\n" % (l, answers.get(l, -1)))
    return answers, y2p, ids  # for debug


  def train_and_submit(self, alg='sklr', srate=0.9):
    ml = top.ml(alg=alg, en_plot=False, save=True)
    ml.train(tmin=0, tmax=0.9)
    filepath = ml.filepath.split('/')[-1].split('.')[0]
    self.predict_file(filepath+'.pkl')
    print("[Submitter] submit file generated in %s @ %s" % (self.submitFile, datetime.now()))


  def ensemble_and_submit(self):
    # [TODO]
    return

#==========================================
#   main
#==========================================
if __name__ == '__main__':  
  cmd = str(sys.argv[1])
  if (len(sys.argv) > 2): cmd2 = str(sys.argv[2])
  if (len(sys.argv) > 3): cmd3 = str(sys.argv[3])

  # [load model & submit]
  if cmd == 'submit':
    sub = submitter(cmd2)
    sub.predict_file(cmd2+'.pkl')
  elif cmd == 'fast_data_layer1':
    sub = submitter(cmd2, en_fast_data=cmd3, sampleFile='train.csv')
    sub.predict_file(cmd2+'.pkl', submit=False)
  elif cmd == 'fast_data_submit':
    sub = submitter(cmd2, en_fast_data=cmd3)
    sub.predict_file(cmd2+'.pkl')
  elif cmd == 'fast_load_submit':
    sub = submitter(cmd2, en_fast_load=True)
    sub.predict_file(cmd2+'.pkl')
  elif cmd == 'train_n_submit':
    # [train & submit]
    submitter('submit_flow').train_and_submit()


