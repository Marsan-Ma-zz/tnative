import sys, random, json, os, re
import numpy as np
import matplotlib.pyplot as plt

from math import log, exp
from sklearn.metrics import roc_curve, auc

from lib import kaggle_auc as kauc

#==========================================
#   flow class
#==========================================  
class grader(object):
  def __init__(self, en_plot=False):
    self.en_plot = en_plot

  #-------------------------
  #   for fun~
  #-------------------------
  def kaggle_score(self, rank, teams=1000, day=30):
    score = 1e5*(rank**-0.75)*log(1+log(teams, 10), 10)*exp(-day/500)
    # print(score)
    return score

  def kaggle_score_curve(self, rank=10, teams=1000, day=30):
    rank_score = []
    teams_score = []
    for r in range(1, 100):
      rank_score.append(self.kaggle_score(r, teams, day))
    for t in range(10, 1000):
      teams_score.append(self.kaggle_score(rank, t, day))
    fig = plt.figure(figsize=(12,4))
    # score to rank
    ax1 = fig.add_subplot(1, 2, 1)
    ax1.plot(rank_score, label='score')
    ax1.set_xlabel('rank')
    ax1.set_ylabel('score')
    ax1.set_title('Kaggle rank to score, while teams=%i' % teams)
    ax1.legend(loc="upper right")
    # score to teams
    ax2 = fig.add_subplot(1, 2, 2)
    ax2.plot(teams_score, label='score')
    ax2.set_xlabel('teams')
    ax2.set_ylabel('score')
    ax2.set_title('Kaggle teams to score, while rank=%i' % rank)
    ax2.legend(loc="upper right")

  #-------------------------
  #   performance
  #-------------------------
  def scan_all_threshold(self, y2p):
    clk = len([y for y, p in y2p if (y > 0)])
    scan = []
    for th in np.arange(1, 0, -0.1):
      imp = len([y for y, p in y2p if (p > th)])
      if (imp == 0): continue   # else will cause divide by 0 error
      tp = len([y for y, p in y2p if ((p >= th) & (y > 0))])
      fp = len([y for y, p in y2p if ((p >= th) & (y == 0))])
      tn = len([y for y, p in y2p if ((p < th) & (y == 0))])
      fn = len([y for y, p in y2p if ((p < th) & (y > 0))])
      hit_rate = float(tp+tn) / len(y2p)
      scan.append({'th': th, 'tp': tp, 'fp': fp, 'tn': tn, 'fn': fn})
      print("th=%.2f, hit_rate=%.5f (tp=%i/fp=%i/tn=%i/fn=%i)" % (th, hit_rate, tp, fp, tn, fn))
    return scan

  def find_pnorm(self, y2p, threshold=1):
    pp = sorted([p for y, p in y2p])
    pth = int(len(pp)*threshold)
    pnorm = pp[pth-1]
    # print "pnorm=%f" % (pnorm)
    if self.en_plot:
      fig = plt.figure(figsize=(6,2.5))
      ax = fig.add_subplot(1, 1, 1)
      ax.plot(pp[:pth])
      ax.set_title("th=%.3f on %f" % (threshold, pnorm))
      ax.set_ylabel('prob %')
      ax.set_xlim([-pth*0.05, pth*1.05])
      plt.show()
    return pnorm

  def auc_curve(self, y2p):
    test, preds = list(zip(*y2p))
    fpr, tpr, thresholds = roc_curve(test, preds)
    roc_auc = auc(fpr, tpr)
    if self.en_plot:
      fig = plt.figure(figsize=(12,4))
      # auc curve
      ax1 = fig.add_subplot(1, 2, 1)
      ax1.plot(fpr, tpr, lw=1, label='ROC (area = %0.5f)' % (roc_auc))
      ax1.plot([0, 1], [0, 1], '--', color=(0.6, 0.6, 0.6), label='Luck')
      ax1.set_xlim([-0.05, 1.05])
      ax1.set_ylim([-0.05, 1.05])
      ax1.set_xlabel('False Positive Rate')
      ax1.set_ylabel('True Positive Rate')
      ax1.set_title('Receiver operating characteristic example')
      ax1.legend(loc="lower right")
      # predict distribution
      ax2 = fig.add_subplot(1, 2, 2)
      yy, pp = list(zip(*y2p))
      yy = sorted(yy)
      pp = sorted(pp)
      ax2.plot(yy, '--', color=(0.6, 0.6, 0.6), label='pred')
      ax2.plot(pp, label='true')
      ax2.set_title("predicted values")
      ax2.set_xlabel('samples')
      ax2.set_ylabel('prob %')
      ax2.legend(loc="upper left")
      plt.show()
    # calculate with kaggle official metric
    actual, posterior = zip(*y2p)
    kaggle_auc = kauc.auc(actual, posterior)
    print('ROC (area = %0.5f) / kaggle AUC = %.5f' % (roc_auc, kaggle_auc))
    return roc_auc



