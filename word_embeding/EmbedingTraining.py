#!/usr/env/bin python
#coding=utf-8

import os
import sys
import random
import collections
import math
import tensorflow as tf
import numpy as np
import pickle as pkl

sys.path.append(".")
sys.path.append("..")
import libs.Configparser

class EmbedingTraning(object):
    def __init__(self, input_file):
        # 基本参数
        self.batch_size = None          # 每批运行的batch大小，这里为不定大小

        # 模型相关参数配置
        self.embeding_dim = None        # 词向量维度
        self.neg_sample_num = None      # 负采样个数
        self.window = None              # 窗口大小
        self.learning_rate = None       # 学习速度
        self.logdir = None              # 日志保存地址
        self.batch_percent = None       # 每次运行训练输入数据占总量的占比
        self.save_path = None           # 模型保存路径
        self.stop_words = []            # 停用词表
        # 加载配置文件
        self.__load_config()

        # 数据相关
        self._input_file = input_file   # 输入文件
        self.sentence_list = []         # 输入query列表（分词后）
        self.word2id = {}               # word=>id的映射
        self.vocab_list = []            # 词表
        self.vocab_size = 0             # 词表大小
        # 读入输入文件
        self.__read_input()

        # 模型训练次数相关
        self.train_words_num = 0        # 训练的单词对数
        self.train_sents_num = 0        # 训练的句子数
        self.train_times_num = 0        # 训练的次数（一次可以有多个句子）

        # train loss
        self.train_loss_records = collections.deque(maxlen=10)  # 保存最近10次的误差
        self.train_loss_k10 = 0

        self.build_graph()
        self.init_op()

    def __read_input(self):
        """
        读取输入文件
        :return:
        """
        fp = open(self._input_file, 'r')
        line = True
        dictTermFreq = {}
        while line:
            line = fp.readline().strip('\r\n')
            # tmp_vocabs = line.replace(" ", "").split("\t")
            # sent_vocabs = []
            # for term in tmp_vocabs:
            #     if term != "" and term not in self.stop_words:
            #         sent_vocabs.append(term)
            sent_vocabs = line.replace(" ", "").split("\t")
            self.sentence_list.append(sent_vocabs)
            for term in sent_vocabs:
                if not dictTermFreq.has_key(term):
                    dictTermFreq[term] = 0
                dictTermFreq[term] += 1
        fp.close()
        # 按词频进行排序，频率从低到高排序，频率高的在tf采样时更容易被抽到
        liSorted = sorted(dictTermFreq.items(), key=lambda x:x[1])
        for idx in range(len(liSorted)):
            term = liSorted[idx][0]
            self.vocab_list.append(term)
            self.word2id[term] = idx
        self.vocab_size = self.vocab_list.__len__()

    def __load_config(self):
        """
        加载配置文件
        :return:
        """
        config_file = "./config/embeding.conf"
        config_parser = libs.Configparser.Parser(config_file)
        # 词向量的维度
        self.embeding_dim = int(config_parser.get("embeding", "embeding_dim"))
        # 负采样的个数
        self.neg_sample_num = int(config_parser.get("embeding", "neg_sample_num"))
        # 窗口大小
        self.window = int(config_parser.get("embeding", "window"))
        # 学习速度
        self.learning_rate = float(config_parser.get("embeding", "learning_rate"))
        # 日志保存地址
        self.logdir = config_parser.get("embeding", "logdir")
        # 每次运行训练输入数据占总量的占比
        self.batch_percent = float(config_parser.get("embeding", "batch_percent"))
        # 模型保存路径
        self.save_path = config_parser.get("embeding", "model_save_path")
        # 停用词保存路径
        stop_words_dic = config_parser.get("embeding", "stop_words")
        with open(stop_words_dic) as fp:
            self.stop_words = [line.strip("\r\n") for line in fp.readlines()]
            fp.close()

    def init_op(self):
        self.sess = tf.Session(graph=self.graph)
        self.sess.run(self.init)
        self.summary_writer = tf.summary.FileWriter(self.logdir, self.sess.graph)

    def build_graph(self):
        self.graph = tf.Graph()
        with self.graph.as_default():
            self.train_inputs = tf.placeholder(tf.int32, shape=[self.batch_size])
            self.train_labels = tf.placeholder(tf.int32, shape=[self.batch_size, 1])

            self.embedding_dict = tf.Variable(
                tf.random_uniform([self.vocab_size,self.embeding_dim],-1.0,1.0)
            )

            self.nce_weight = tf.Variable(tf.truncated_normal([self.vocab_size, self.embeding_dim],
                                                  stddev=1.0/math.sqrt(self.embeding_dim)))
            self.nce_biases = tf.Variable(tf.zeros([self.vocab_size]))

            # 获取输入序列的向量
            embs = tf.nn.embedding_lookup(self.embedding_dict, self.train_inputs)

            # 得到NCE损失
            self.loss = tf.reduce_mean(
                tf.nn.nce_loss(
                    weights = self.nce_weight,
                    biases = self.nce_biases,
                    labels = self.train_labels,
                    inputs = embs,
                    num_sampled = self.neg_sample_num,
                    num_classes = self.vocab_size
                )
            )

            # tensorboard 相关
            tf.summary.scalar('loss', self.loss)  # 让tensorflow记录参数

            # 梯度下降
            #self.train_op = tf.train.GradientDescentOptimizer(learning_rate=self.learning_rate).minimize(self.loss)
            self.train_op = tf.train.AdamOptimizer().minimize(self.loss)

            # 计算和输入词语的相似度
            self.test_inputs = tf.placeholder(tf.int32, shape=[None])
            # # 词向量求模
            # vec_l2_model = tf.sqrt(
            #     tf.reduce_sum(tf.square(self.embedding_dict), 1, keep_dims=True)
            # )
            # # 整体模
            # avg_l2_model = tf.reduce_mean(vec_l2_model)
            # # 向量归一化
            # self.normaled_embedding_dict = self.embedding_dict / vec_l2_model
            self.normal_embeding()

            test_embs = tf.nn.embedding_lookup(self.normaled_embedding_dict, self.test_inputs)   # 测试的输入向量
            # 计算测试词语和词向量字典的相似度
            self.similarity = tf.matmul(test_embs, self.normaled_embedding_dict, transpose_b=True)

            # 变量初始化
            self.init = tf.global_variables_initializer()

            self.merged_summary_op = tf.summary.merge_all()

            self.saver = tf.train.Saver()

    def normal_embeding(self):
        # 词向量求模
        vec_l2_model = tf.sqrt(
            tf.reduce_sum(tf.square(self.embedding_dict), 1, keep_dims=True)
        )
        # 整体模
        avg_l2_model = tf.reduce_mean(vec_l2_model)
        # 向量归一化
        self.normaled_embedding_dict = self.embedding_dict / vec_l2_model

    def train(self):
        # 随机抽样一批sentence做训练
        sent_samples = random.sample(self.sentence_list, int(self.batch_percent*self.sentence_list.__len__()))
        batch_inputs = []
        batch_labels = []
        for sent_vocabs in sent_samples:
            sent_length = sent_vocabs.__len__()
            for idx in range(len(sent_vocabs)):
                start = max(0, idx-self.window)
                stop = mim(sent_length, idx+self.window+1)
                for i in range(start, stop):
                    if i == idx:
                        continue
                    input_id = self.word2id.get(sent_vocabs[idx])
                    label_id = self.word2id.get(sent_vocabs[i])
                    if not input_id or not label_id:
                        continue
                    batch_inputs.append(input_id)
                    batch_labels.append(label_id)
        if batch_inputs == []:
            return
        batch_inputs = np.array(batch_inputs, dtype=np.int32)
        batch_labels = np.array(batch_labels, dtype=np.int32)
        batch_labels = np.reshape(batch_labels,[batch_labels.__len__(),1])
        feeddict = {
            self.train_inputs: batch_inputs,
            self.train_labels: batch_labels
        }

        _, loss_val, summary_str = self.sess.run([self.train_op,self.loss,self.merged_summary_op], feed_dict=feeddict)

        # 记录train loss
        self.train_loss_records.append(loss_val)
        self.train_loss_k10 = np.mean(self.train_loss_records)

        # 记录训练次数
        self.train_sents_num += sent_samples.__len__()          # 训练的sentence数量
        self.train_words_num += batch_inputs.__len__()          # 训练的词语对数量
        self.train_times_num += 1                               # 总训练次数

        if self.train_times_num % 100 == 0:
            self.summary_writer.add_summary(summary_str, self.train_sents_num)
            print "train time=%d, train sentences number=%d, train words number=%d, loss=%s" % (
                self.train_times_num, self.train_sents_num, self.train_words_num, self.train_loss_k10
            )

    def save_model(self):
        if not os.path.exists(self.save_path):
            os.mkdir(self.save_path)

        # 记录模型各参数
        model = {}
        var_names = ['vocab_size',      # int       model parameters
                     'vocab_list',      # list
                     'learning_rate',   # float
                     'word2id',         # dict
                     'embeding_dim',   # int
                     'logdir',          # str
                     'window',         # int
                     'neg_sample_num',     # int
                     'train_words_num', # int       train info
                     'train_sents_num', # int
                     'train_times_num', # int
                     'train_loss_records',  # int   train loss
                     'train_loss_k10',  # int
                     ]
        for var in var_names:
            model[var] = eval('self.'+var)

        param_path = os.path.join(self.save_path,'params.pkl')
        if os.path.exists(param_path):
            os.remove(param_path)
        with open(param_path,'wb') as f:
            pkl.dump(model,f)

        self.normal_embeding()
        normaled_embedding_dict = self.sess.run(self.normaled_embedding_dict).tolist()
        with open(os.path.join(self.save_path, 'word2vec.dic'), 'w') as wp:
            for word, id in self.word2id.items():
                print >> wp, "%s\t%s" % (word, "\t".join([str(elem) for elem in normaled_embedding_dict[id]]))
            wp.close()

        # 记录tf模型
        tf_path = os.path.join(self.save_path, 'tf_vars')
        if os.path.exists(tf_path):
            os.remove(tf_path)
        self.saver.save(self.sess, tf_path)

    def get_test_id(self, test_words):
        test_ids = []
        for word in test_words:
            test_ids.append(self.word2id.get(word))
        return test_ids

    def cal_similarity(self, test_ids, top_k=5):
        test_ids = np.array(test_ids, dtype=np.int32)
        sim_matrix = self.sess.run([self.similarity], feed_dict={self.test_inputs: test_ids})[0]
        test_words = []
        nearest_words = []
        for i in range(len(test_ids)):
            test_words.append(self.vocab_list[test_ids[i]])
            nearst_ids = (-sim_matrix[i,:]).argsort()[1:top_k+1]
            nearest_word_single = [self.vocab_list[idx] for idx in nearst_ids]
            nearest_words.append(nearest_word_single)
        for word, nearest_word_single in zip(test_words, nearest_words):
            print "%s\t%s" % (word, ",".join(nearest_word_single))


if __name__ == "__main__":
    input_file = sys.argv[1]
    train_handler = EmbedingTraning(input_file)
    test_words = ["小米", "发货", "手机", "订单"]
    test_ids = train_handler.get_test_id(test_words)
    for count in range(1, 10001):
        train_handler.train()
        if count % 100 == 0:
            train_handler.cal_similarity(test_ids)
        if count % 10000 == 0:
            # 保存模型
            train_handler.save_model()











