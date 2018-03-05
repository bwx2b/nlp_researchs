#!/usr/env/bin python
#coding=utf-8

import logging

import jieba

class Nlp(object):
    def __init__(self):
        self.dict_path = "~/workspace/Dict"

    def load_dict(self):
        jieba_userdict = "%s/newwords.dic" % self.dict_path
        if not os.path.exists(jieba_userdict):
            logging.warn("jieba user dict %s not exists" % jieba_userdict)
        else:
            jieba.load_userdict(jieba_userdict)
            logging.info("load jieba user dict %s done" % jieba_userdict)

    def wordseg(self, query):
        return [word.encode("utf-8") for word in jieba.cut(query)]