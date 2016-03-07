import sys, json, re, requests
import urllib.parse
import xml.etree.ElementTree as ET
import multiprocessing as mp
pool_size = int(mp.cpu_count())

from lxml import html 
from bs4 import BeautifulSoup as bs

#----------------------------------
#   scraper
#----------------------------------
class scraper(object):
  def __init__(self):
    self.url_validator = re.compile(
      r'^(?:http|ftp)s?://' # http:// or https://
      r'(?:(?:[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?\.)+(?:[A-Z]{2,6}\.?|[A-Z0-9-]{2,}\.?)|' #domain...
      r'localhost|' #localhost...
      r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})' # ...or ip
      r'(?::\d+)?' # optional port
      r'(?:/?|[/?]\S+)$', re.IGNORECASE
    )

  #-------------------------
  #   tasks
  #-------------------------
  def validate_url(self, url):
    ans = True if re.match(self.url_validator, url) else False
    return ans


  #-------------------------
  #   site/url quality
  #-------------------------
  def domain_categorize(self, domain):
    url = 'http://sitereview.bluecoat.com/sitereview.jsp#/?search='+domain
    r = Render(url)  
    result = r.frame.toHtml()
    soup = bs(result)
    

  def alexa_rank(self, url):
    if not self.validate_url(url):
      print("[alexa_rank] skip, not url: "+url)
      return
    else:
      resp = requests.get("http://data.alexa.com/data?cli=10&dat=s&url="+url)
      xml_tree = ET.fromstring(resp.text)
      try:
        score = xml_tree[1]
        results = {
          'url'       : url,
          'rank'      : int(score[1].attrib['RANK']) if len(score) > 1 else None,
          'delta'     : int(score[2].attrib['DELTA']) if len(score) > 2 else None,
          'loc_name'  : score[3].attrib['NAME'] if len(score) > 3 else None,
          'loc_rank'  : int(score[3].attrib['RANK']) if len(score) > 3 else None,
        }
        return results
      except Exception as e:
        print("[alexa_rank] error: ", resp.text)
        return


  def batch_get_alexa_rank(self, urls):
    mp_pool = mp.Pool(pool_size)
    processes = []
    results = {}
    for u in urls:
      p = mp_pool.apply_async(self.alexa_rank, (u,))
      processes.append(p)
    for p in processes:
      r = p.get() #timeout=10)
      if r: results[r['url']] = r
    return results


  def batch_get_fb_info(self, urls):
    # urls = ['http://www.google.com', 'http://www.piposay.com', 'not a url', 'still not url', 'http://edition.cnn.com/', 'www.bbc.co.uk']
    fields = "url, share_count, like_count, comment_count, total_count, click_count"
    graph_api = 'https://graph.facebook.com/fql'
    cmd = "%s?q=SELECT %s FROM link_stat WHERE url in (%s)" % (graph_api, fields, ",".join(["'%s'" % s for s in urls]))
    resp = requests.get(cmd)
    results = json.loads(resp.text)
    # print(results)
    results = results.get('data')
    if not results: 
      print("-"*40, resp.text, "-"*40)
      for u in urls: print(urllib.parse.unquote(u))
    # print(len(results), results[:5])
    return results







# #----------------------------------
# #   scrap content loaded by javascript
# #----------------------------------
# from PyQt4.QtGui import *  
# from PyQt4.QtCore import *  
# from PyQt4.QtWebKit import *  
# class Render(QWebPage):  
#   def __init__(self, url):  
#     self.app = QApplication(sys.argv)  
#     QWebPage.__init__(self)  
#     self.loadFinished.connect(self._loadFinished)  
#     self.mainFrame().load(QUrl(url))  
#     self.app.exec_()  
  
#   def _loadFinished(self, result):  
#     self.frame = self.mainFrame()  
#     self.app.quit()  

