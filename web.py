#-*- coding:utf-8 -*-

# web framework + server
import os, bottle, cherrypy, imp, timeit, copy, json, pickle, socket, getpass
import numpy as np
import html

from bs4 import BeautifulSoup as bs
from bottle import route, run, template, response
from bottle import get, post, request, debug
from bottle import static_file

# mylib
from lib import schema as db

root_path = "/home/marsan/workspace/tnative"

#===========================================
#   API
#===========================================
@get('/')
def index():
  articles = db.articles.objects(isad__ne=None).limit(100)
  return template('%s/views/home.tpl' % root_path, articles=articles)

@get('/isad')
def isad():
  articles = db.articles.objects(isad=True).limit(100)
  return template('%s/views/home.tpl' % root_path, articles=articles)

@get('/notad')
def notad():
  articles = db.articles.objects(isad=False).limit(100)
  return template('%s/views/home.tpl' % root_path, articles=articles)

@get('/empty')
def notad():
  articles = db.articles.objects(status='empty').limit(100)
  return template('%s/views/home.tpl' % root_path, articles=articles)

@get('/bad')
def notad():
  articles = db.articles.objects(status='bad').limit(100)
  return template('%s/views/home.tpl' % root_path, articles=articles)

@get('/debug/<name>')
def debug(name):
  articles = db.articles.objects(debug=name).limit(100)
  return template('%s/views/home.tpl' % root_path, articles=articles)


# @get('/favicons/<filename:re:.*\.(jpg|png|gif|ico)>')
@get('/favicons/<filename>')
def favicons(filename):
  return static_file(filename, root=root_path+'/assets/favicons/')

@get('/articles/<id>')
def show_article(id):
  doc = db.articles.objects(id=id).first()
  text = doc.html
  if (doc.status == 'bad'): 
    # text = bs(text).body.p.text
    text = ''.join(text.split())
    # text = html.escape(text)
    text = html.unescape(text)
  return text


#===========================================
#   Server
#===========================================
if __name__ == '__main__':
  hostname = socket.gethostname()
  if (getpass.getuser() == 'marsan'):
    bottle.debug(True)
    bottle.run(server='cherrypy', host='10.0.0.6', port=6804, reloader=True, debug=True)
  else:
    bottle.run(server='cherrypy', host='10.0.0.6', port=6804, reloader=True, debug=True)
  