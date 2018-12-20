#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2018-04-23 19:46:18
# @Author  : AlexTang (1174779123@qq.com)
# @Link    : http://t1174779123.iteye.com
# @Description : 封装模型，融合规则

import os
import re
import conf
import codecs
import random
from itertools import groupby
from nlp_util_py3 import NLPUtil
#from entity_api import get_types
from rule_util import rule_predict
from fasttext_util import FasttextClassifier
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
#from log import g_log_inst as logger


class IntentClassify(object):
    """docstring for IntentClassify"""
    def __init__(self, model_info, use_rule=True, 
        token_func=lambda x:FasttextClassifier.fasttext_tokenize(x['text']), 
        pretrain='', quantize=False):
        self.use_rule = use_rule
        self.token_func = token_func
        self.pretrain = pretrain
        self.quantize = quantize
        self.models = []
        self.train_x = []
        self.train_y = []
        if isinstance(model_info, (str, unicode)) and os.path.isdir(model_info):
            for model_file in os.listdir(model_info):
                model_fpath = os.path.join(model_info, model_file)
                if os.path.isfile(model_fpath):
                    self.models.append(FasttextClassifier.load_model(model_fpath))
            print 'load %s models from %s' % (len(self.models), model_info)
        elif isinstance(model_info, list):
            for param in model_info:
                self.models.append(FasttextClassifier(*param))
                
    def get_sentence_vector(self, sent):
        sen_vec = [m.classifier_.get_sentence_vector(sent)  for m in self.models]
        return sen_vec

    
    def cut_text(self, data_fpath):
        keys = ['label', 'text']
        with codecs.open(data_fpath, 'r', 'utf-8') as f:
            infos = [x.strip('\n').split('\t') for x in f]
            infos = [{keys[i]:x[i] for i in range(len(keys))} for x in infos]
        random.shuffle(infos)
        tmp_fpath = data_fpath+'.tmp'    
        with codecs.open(tmp_fpath, 'w', 'utf-8') as f:
            _ = map(lambda x:f.write('__label__%s\t%s\n'%(
                        x['label'],self.token_func(x))),
                    infos)
            
            
    def train(self, data_fpath, model_path):
        keys = ['label', 'text']
        with codecs.open(data_fpath, 'r', 'utf-8') as f:
            infos = [x.strip('\n').split('\t') for x in f]
            infos = [{keys[i]:x[i] for i in range(len(keys))} for x in infos]
        random.shuffle(infos)
        tmp_fpath = data_fpath+'.tmp'
        with codecs.open(tmp_fpath, 'w', 'utf-8') as f:
            _ = map(lambda x:f.write('__label__%s\t%s\n'%(
                        x['label'],self.token_func(x))),
                    infos)
        if not os.path.exists(model_path):
            os.mkdir(model_path)
        _ = map(lambda x:x.train(
            tmp_fpath, 
            os.path.join(
                model_path,
                'fasttext_%s_%s_%s_%s'%(
                    x.word_ngrams,
                    x.epoch,
                    x.dim,
                    x.lr)),
            self.pretrain,
            self.quantize),
            self.models)

    def fit(self, train_x, train_y):
        self.train_x = train_x
        self.train_y = train_y
        _ = map(lambda x:x.fit(train_x, train_y, pretrainedVectors=self.pretrain), self.models)
        print 'fit %s models success!' % (len(self.models))

    def predict(self, text,  k=1):
        label, proba = self.predict_proba(text, k)
        return label

    def predict_proba(self, data, k=1):
        # rule first
        if self.use_rule:
            label = rule_predict(data['text'])
            if label:
                return label, 1.0
        
        # stop-words only text return chat
        fasttext_text = self.token_func(data)
        # print fasttext_text
        if not fasttext_text:
            return u'无意图', 1.0
        # print text
        label, proba = self.model_predict_mutil(fasttext_text)
        # print 'label before return', label
        # print 'predict %s\t%s\t%s\t%s' % (data['text'], fasttext_text, label, proba)
        return label, proba

    def model_predict_mutil(self, text):
        mutil_results = [self.model_predict(m, text, 1) for m in self.models]
        # print 'mutil_results: ',mutil_results
        label, proba = self.find_best_label(mutil_results)
        return label.replace('__label__',''), proba

    def model_predict(self, clf, text, k=1):
        labels, probas = clf.predict_proba([text], k)
        # print('in intent_classify:', labels, probs)
        return labels[0][0], probas[0][0]

    @classmethod
    def find_best_label(cls, mutil_results):
        '''
        # return highest score label
        mutil_results.sort(key=lambda x:x[1], reverse=True)
        result = mutil_results[0][0]
        return result
        '''
        # return most label
        mutil_results.sort(key=lambda x:x[0])
        label_results = [(l, list(group)) for l, group in groupby(mutil_results, lambda x:x[0])]
        max_cnt = max([len(x[1]) for x in label_results])
        # find candicate labels first
        cand_labels = [x for x in label_results if len(x[1])==max_cnt]
        cand_labels.sort(key=lambda x:max([y[1] for y in x[1]]), reverse=True)
        label = cand_labels[0][0]
        proba = max([x[1] for x in cand_labels[0][1]])
        return label, proba
        
    def stat_label(self, x, y, y_pre):
        recall = recall_score(y, y_pre, average='micro')
        precision = precision_score(y, y_pre, average='micro')
        f1_score = 2 * recall * precision / (recall + precision)
        #logger.get().info('model_num=%s, f1_score=%f', len(self.models), f1_score)
        # stat label recall
        y_comb = zip(y, y_pre)
        y_comb.sort(key=lambda x:x[0])
        label_results = {'all':{'train_cnt':len(self.train_y),
                                'true_cnt':len(y), 
                                'recall':recall, 
                                'pre_cnt':len(y), 
                                'prec':precision,
                                'f1':f1_score}}
        for label, group in groupby(y_comb, lambda x:x[0]):
            group = list(group)
            g_y = [x[0] for x in group]
            g_pre = [x[1] for x in group]
            label_recall = len([x for x in g_pre if x==label])/float(len(group))
            # label_recall = recall_score(g_y, g_pre, average='macro')
            label_results[label]= {
                'train_cnt':len([x for x in self.train_y if label in x]),
                'true_cnt':len(group), 
                'recall':label_recall, 
                'pre_cnt':0,
                'prec':0.0
            }
            #logger.get().info('label: %s, cnt: %s, recall: %s',
                #label, len(group), label_recall)
        # stat label prec
        y_comb.sort(key=lambda x:x[1])
        for label, group in groupby(y_comb, lambda x:x[1]):
            group = list(group)
            g_y = [x[0] for x in group]
            g_pre = [x[1] for x in group]
            label_prec = len([x for x in g_y if x==label])/float(len(group))
            # label_prec = precision_score(g_y, g_pre, average='macro')
            if label not in label_results:
                continue
            label_results[label]['pre_cnt'] = len(group)
            label_results[label]['prec'] = label_prec
        # print '-'*80
        # print 'model_num=%s, recall=%f, Prec=%f' % (len(self.models), recall, precision)
        # print '\n'.join(['%s,%s,%s,%s,%s'%(k,v['true_cnt'],v['recall'],v['pre_cnt'],v['prec']) for k,v in label_results.items()])
        return label_results


if __name__ == '__main__':
    #logger.start('../log/intent_classify.log', __name__, 'DEBUG')
    '''
        predict code, load pre-trained models
    '''
    # intent_classify = IntentClassify('../model/')
    # print 'mdl predict: %s, %s' % (intent_classify.predict_proba(u'baby一下网络歌曲', [{'text': '', 'intent':'chat'}]))
    # print 'mdl predict: %s, %s' % (intent_classify.predict_proba(u'遭遇', [{'text': '', 'intent':'chat'}]))
    # print u'播放dj版的音乐', intent_classify.predict(u'播放dj版的音乐')
    # print intent_classify.predict(u'取消', [{'text': u'播放音乐', 'intent':'music.play'}])

    '''
        train model
    '''
    # model_pretrain_char_1 = IntentClassify(
    #     [(2, 10, 100, 0.8)], use_rule=False, 
    #     token_func=lambda x:FasttextClassifier.fasttext_tokenize(x, char=True),
    #     pretrain='../conf/fasttext.model.min30_ngram5_d100_it20.vec',
    #     quantize=True)
    # # model_pretrain_char_1.train('../data/kangmei_data/kangmei_data_0609.txt', '../model/app8/')
    
    # model_pretrain_char_2 = IntentClassify(
    #     [(2, 10, 100, 0.6)], use_rule=False, 
    #     token_func=lambda x:FasttextClassifier.fasttext_tokenize(x, char=True),
    #     pretrain='../conf/fasttext.model.min30_ngram5_d100_it20.vec',
    #     quantize=True)
    # # model_pretrain_char_2.train('../data/kangmei_data/kangmei_comfirm_0609.txt', '../model/app7/')
    
    # model_pretrain_char_3 = IntentClassify(
    #     [(2, 10, 100, 0.2)], use_rule=False, 
    #     token_func=lambda x:FasttextClassifier.fasttext_tokenize(x['text'], char=True),
    #     pretrain='../conf/fasttext.model.min30_ngram5_d100_it20.vec',
    #     quantize=False)
    # model_pretrain_char_3.train('../data/baole_data/baole_fasttext.txt', '../model/baole/')
    

    
    print add_entity_idx_tokenize({'text': u'黎宇我要玩中国象棋'})



    