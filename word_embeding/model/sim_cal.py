#!/usr/env/bin python
#coding=utf-8

import os
import sys
import math

emb_dict = {}

def init_emb(filename):
    with open(filename, 'r') as fp:
        liLines = fp.readlines()
        fp.close()
    for line in liLines:
        liTmp = line.strip('\r\n').split('\t')
        if len(liTmp) < 201:
            continue
        term = liTmp[0]
        vec = [float(elem) for elem in liTmp[1:]]
        emb_dict[term] = vec

def cosine(vec1, vec2):
    vv = sum([elem*elem for elem in vec1])
    dd = sum([elem*elem for elem in vec2])
    vd = sum([elem1*elem2 for elem1, elem2 in zip(vec1, vec2)])
    return vd/math.sqrt(vv*dd)

def get_top_sim(term, n=3):
    vec = emb_dict[term]
    dictSim = {}
    for key, value in emb_dict.items():
        if key == term:
            continue
        dictSim[key] = cosine(vec, value)
    liSorted = sorted(dictSim.items(), key=lambda x:x[1], reverse=True)
    return liSorted[0:n]

if __name__ == "__main__":
    init_emb("word2vec.dic")
    print "加载词向量字典成功"
    while True:
        type_ = raw_input("请输入类型 top3-计算单词最近的三个query | sim-计算两个单词的相似度\n")
        if type_ == "exit":
            sys.exit(0)
        elif type_ not in ["sim", "top3"]:
            print "无效类型"
            continue
        elif type_ == "sim":
            term_1 = raw_input("请输入第一个单词:\n")
            if term_1 == "exit":
                sys.exit(0)
            elif not emb_dict.has_key(term_1):
                print "%s不存在, 请重新输入" % term_1
                continue
            term_2 = raw_input("请输入第二个单词:\n")
            if term_2 == "exit":
                sys.exit(0)
            elif not emb_dict.has_key(term_2):
                print "%s不存在, 请重新输入" % term_2
                continue
            print "%s\t%s\t%s" % (term_1, term_2, cosine(emb_dict[term_1], emb_dict[term_2]))
        elif type_ == "top3":
            term = raw_input("请输入单词:\n")
            if not emb_dict.has_key(term):
                print "%s不存在, 请重新输入" % term
                continue
            liRes = get_top_sim(term)
            for elem in liRes:
                print "%s\t%s" % (elem[0], str(elem[1]))

            
