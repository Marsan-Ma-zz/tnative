#-*- coding:utf-8 -*-
#==================================
#   [MongoEngine] http://mongoengine-odm.readthedocs.org/tutorial.html
#==================================
from mongoengine import *
from pymongo import MongoClient
import datetime

# connect('maxminer_test', host='localhost', port=12999)
# connect('tnative', host='10.0.0.4', port=12999)
connect('tnative_omega', host='10.0.0.6', port=12999)

#==================================
#   feature hashed samples
#==================================
class samples(Document):
  mode              = StringField()
  status            = StringField()
  fid               = IntField(required=True)
  label             = BooleanField()
  rand              = FloatField()
  features          = ListField()
  feature_size      = IntField()
  meta = {
    'indexes': [
      'mode',
      'fid',
      'label',
      'rand',
      # 'feature_size',
    ]
  }

#==================================
#   Samples
#==================================
class articles(Document):
  # raw data
  status            = StringField(default='normal')   # normal/bad/empty
  fid               = IntField(required=True, unique=True)
  isad              = BooleanField()
  rand              = FloatField()
  html              = StringField()
  utime             = DateTimeField(default=datetime.datetime.now)
  processed         = BooleanField(default=False)
  # newspaper features
  summary           = StringField()
  keywords          = ListField()
  # goose features
  title             = StringField()
  cleaned_text      = StringField()
  tags              = ListField()
  opengraph         = DictField()
  link_stats        = DictField()
  link_factors      = ListField()
  tweets            = ListField()
  top_image         = StringField()
  movies            = ListField()
  publish_date      = DateTimeField()
  meta_site_name    = StringField()
  meta_lang         = StringField()
  meta_description  = StringField()
  meta_keywords     = ListField()
  canonical_link    = StringField()
  domain            = StringField()
  authors           = ListField()
  # aggregated features
  wday              = IntField()
  ihour             = IntField()
  # ner
  ner_facility      = ListField()
  ner_gpe           = ListField()
  ner_location      = ListField()
  ner_org           = ListField()
  ner_person        = ListField()
  ner_date          = ListField()
  # [author stat]
  is_pure_ad_authors        = BooleanField()
  is_prefer_ad_authors      = BooleanField()
  is_prefer_nonad_authors   = BooleanField()
  is_pure_nonad_authors     = BooleanField()
  # [domain stat]
  is_pure_ad_domains        = BooleanField()
  is_prefer_ad_domains      = BooleanField()
  is_prefer_nonad_domains   = BooleanField()
  is_pure_nonad_domains     = BooleanField()
  # [link stat]
  has_pure_ad_links         = BooleanField()
  has_prefer_ad_links       = BooleanField()
  has_pure_nad_links        = BooleanField()
  has_prefer_nad_links      = BooleanField()
  # [social counts]
  fb_click_count            = IntField()
  fb_comment_count          = IntField()
  fb_like_count             = IntField()
  fb_share_count            = IntField()
  fb_total_count            = IntField()
  # [lang]
  lang_adrate_group         = StringField()
  # [doc2vec]
  bow                       = ListField()
  doc2vec                   = ListField()
  tfidf_vec                 = ListField()   # 1000 dim
  # tfidf_vec_svd             = ListField()   # 50 dim
  # [extra]
  lines                     = IntField()
  spaces                    = IntField()
  tabs                      = IntField()
  braces                    = IntField()
  brackets                  = IntField()
  quesmarks                 = IntField()
  exclamarks                = IntField()
  words                     = IntField()
  # [html_tags stat]
  cnt_a                     = IntField()
  cnt_article               = IntField()
  cnt_b                     = IntField()
  cnt_blockquote            = IntField()
  cnt_code                  = IntField()
  cnt_em                    = IntField()
  cnt_form                  = IntField()
  cnt_h1                    = IntField()
  cnt_h2                    = IntField()
  cnt_h3                    = IntField()
  cnt_h4                    = IntField()
  cnt_h5                    = IntField()
  cnt_h6                    = IntField()
  cnt_hr                    = IntField()
  cnt_iframe                = IntField()
  cnt_img                   = IntField()
  cnt_input                 = IntField()
  cnt_li                    = IntField()
  cnt_ol                    = IntField()
  cnt_p                     = IntField()
  cnt_section               = IntField()
  cnt_select                = IntField()
  cnt_small                 = IntField()
  cnt_strike                = IntField()
  cnt_strong                = IntField()
  cnt_table                 = IntField()
  cnt_textarea              = IntField()
  cnt_ul                    = IntField()
  cnt_video                 = IntField()
  #
  cnt_at                    = IntField()
  cnt_blog                  = IntField()
  cnt_jpg                   = IntField()
  cnt_admin                 = IntField()
  cnt_click                 = IntField()
  cnt_feed                  = IntField()
  # 
  rto_html2txt              = FloatField()
  cnt_script                = IntField()
  cnt_style                 = IntField()
  cnt_arrow                 = IntField()
  cnt_slash                 = IntField()
  cnt_backslash             = IntField()
  cnt_meta                  = IntField()
  cnt_wp_content            = IntField()
  # 
  cnt_dicts                 = DictField()
  cnt_bows                  = DictField()
  # [indicator]
  debug                     = StringField()
  empty                     = BooleanField(default=False)
  # [ensemble_model]
  ensemble_0                = DictField()
  ensemble_1                = DictField()
  ensemble_2                = DictField()
  ensemble_3                = DictField()
  ensemble_4                = DictField()
  # [deprecated]
  text              = StringField()
  images            = ListField()
  html_cnt          = IntField()
  cleaned_text_cnt  = IntField()
  links             = DictField()
  meta = {
    'indexes': [
      'fid',
      'isad',
      'rand',
      'utime',
      'processed',
      'domain',
      'status',
      'debug',
      #
      'title',
      ('isad', 'rand'),
    ]
  }

  