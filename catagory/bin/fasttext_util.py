#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2018-04-12 19:53:51
# @Author  : AlexTang (1174779123@qq.com)
# @Link    : http://t1174779123.iteye.com
# @Description : 

import os
import codecs
from datetime import datetime
import fastText as fasttext
# import fasttext
import random
from nlp_util_py3 import NLPUtil
from itertools import groupby
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
#from log import g_log_inst as logger


class FasttextClassifier(BaseEstimator, ClassifierMixin):
    """
    Fasttext classifier.
    """
    def __init__(self, word_ngrams=1, epoch=10, dim=100, lr=0.3,
                 output_dir="./tmp_bin/", bucket=200000):
        self.word_ngrams = word_ngrams
        self.epoch = epoch
        self.dim = dim
        self.lr = lr
        self.output_dir = output_dir
        self.bucket = bucket
        self.classifier = None

        if not os.path.isdir(output_dir):
            os.makedirs(output_dir)

    @classmethod
    def load_model(cls, model_fpath):
        clf = FasttextClassifier()
        clf.classifier_ = fasttext.load_model(model_fpath)
        print 'load model from %s success!' % (model_fpath)
        return clf

    @classmethod
    def fasttext_tokenize(cls, text, char=True, normalize=True, filter_stop_word=False):
        uc_tokens = [u'http_t', u'email_t', u'date_t', u'time_t', u'phone_t', u'int_t', u'float_t']
        token = NLPUtil.tokenize_via_jieba(text, normalize, filter_stop_word)
        if char:
            char_str = ' '.join([x if x in uc_tokens else ' '.join(x) for x in token])
        else:
            char_str = ' '.join(token)
        # print char_str
        return char_str

    def train(self, input_file, output_file, pretrain='', quantize=False):
        file_name = os.path.basename(input_file)
        bin_file  =  output_file + '.bin'
        print 'word_ngrams:%s, epoch:%s, dim:%s, lr:%s, Model saved path is: %s' %(
            self.word_ngrams, self.epoch, self.dim, self.lr, bin_file)
        print "Start training.\n"
        classifier = fasttext.train_supervised(bucket=20000,
                                         input=input_file,
                                         wordNgrams=self.word_ngrams,
                                         pretrainedVectors=pretrain,
                                         epoch=self.epoch,
                                         dim=self.dim,
                                         lr=self.lr,
                                         thread=4)
        self.classifier = classifier
        if quantize:
            self.prune_model(classifier, input_file, output_file)
        return classifier
    
    def prune_model(self, model, input_file, output_file):
        # print dir(model)
        model.quantize(input=input_file, qnorm=True, retrain=True, cutoff=10000)
        model.save_model(output_file)

    def fit(self, X=None, y=None, **fit_params):
        try:
            pid = os.getpid()
            time_str = datetime.strftime(datetime.now(), "%Y-%m-%d_%H-%M-%S")
            input_file_tmp = os.path.join(self.output_dir,
                                          "train_file_tmp_%s_%s" % (pid, time_str))
            if os.path.isfile(input_file_tmp):
                os.remove(input_file_tmp)
            with codecs.open(input_file_tmp, 'w', 'utf-8') as f:
                for idx, text in enumerate(X):
                    f.write("%s\t%s\n" % ('__label__' + y[idx].replace('__label__',''), text))

            output_file = "%s_%s_%s" % (os.path.join(self.output_dir, "model"), pid, time_str)
            output_file_bin = output_file + ".bin"
            if os.path.isfile(output_file_bin):
                os.remove(output_file_bin)
            # pretrained-word vectors
            pretrainedVectors = fit_params['pretrainedVectors'] if 'pretrainedVectors' in fit_params else ''
            if pretrainedVectors:
                print 'pretrainedVectors: %s ' % (pretrainedVectors)
            print 'save model to %s.' % (output_file_bin)
            
            self.classifier_ = fasttext.train_supervised(bucket=20000,
                                         input=input_file_tmp,
                                         wordNgrams=self.word_ngrams,
                                         pretrainedVectors=pretrainedVectors,
                                         epoch=self.epoch,
                                         dim=self.dim,
                                         lr=self.lr,
                                         thread=4)
            print '------'
            self.classifier = self.classifier_
            if self.quantize:
                self.prune_model(self.classifier_, input_file_tmp, output_file)
            if os.path.isfile(input_file_tmp):
                os.remove(input_file_tmp)
            if os.path.isfile(output_file_bin):
                os.remove(output_file_bin)
        except Exception as e:
            print 'FasttextCV failed, errmsg=%s' % (e)
            #logger.get().warn('FasttextCV failed, errmsg=%s' % (e))
        return self

    def predict(self, X, y=None, k=1):
        try:
            getattr(self, "classifier_")
        except AttributeError as e:
            #logger.get().warn('FasttextGridsearch failed, errmsg=%s' % (e))
            raise RuntimeError("You must train classifier before predicting data!")
        labels, probas = self.predict_proba(X, k=k)
        return labels 

    def predict_proba(self, X, y=None, k=1):
        try:
            getattr(self, "classifier_")
        except AttributeError as e:
            #logger.get().warn('Fasttext predict_proba failed, errmsg=%s' % (e))
            raise RuntimeError("You must train classifier before predicting data!")
        # labels, probs = self.classifier_.predict(X, k=k)
        # print('in fasttext_util:',labels, probs)
        if isinstance(X, (str,unicode)):
            labels, probas = self.classifier_.predict([X], k=k)
            if k == 1:
                # print 'try to predict %s ' % (X)
                return labels[0][0], probas[0][0]
            return labels[0], probas[0]
        else:
            # print X
            labels, probas = self.classifier_.predict(X, k=k)
            
        self.classifier = self.classifier_
        return labels, probas

    def score(self, X, y, sample_weight=None):
        try:
            y_pre = self.predict(X)
            y_pre = [x[0].replace(self.classifier_.label_prefix, "") for x in y_pre]
            y = [x.replace(self.classifier_.label_prefix, "") for x in y]
            accuracy = accuracy_score(y, y_pre)
            param_dict = {'word_ngrams': self.word_ngrams, 'epoch':self.epoch, 'dim':self.dim, 'lr': self.lr}
            #logger.get().info('Params=%s, Acc=%f' % (str(param_dict), accuracy))
            print 'Params=%s, Acc=%f' % (str(param_dict), accuracy)
            return accuracy
        except Exception as e:
            print 'FasttextGridsearch failed, errmsg=%s' % (e)
            #logger.get().warn('FasttextGridsearch failed, errmsg=%s' % (e))

    def score_label(self, X, y):
        y_pre = self.predict(X)
        y_pre = [x[0].replace(self.classifier_.label_prefix, "") for x in y_pre]
        y = [x.replace(self.classifier_.label_prefix, "") for x in y]
        return self.stat_label(X, y, y_pre)

    def stat_label(self, x, y, y_pre):
        recall = recall_score(y, y_pre, average='macro')
        precision = precision_score(y, y_pre, average='macro')
        f1_score = 2 * recall * precision / (recall + precision)
        param_dict = {'word_ngrams': self.word_ngrams, 'epoch':self.epoch, 'dim':self.dim, 'lr': self.lr}
        #logger.get().info('Params=%s, f1_score=%f' % (str(param_dict), f1_score))
        # stat label recall
        y_comb = zip(y, y_pre)
        y_comb.sort(key=lambda x:x[0])
        label_results = {'all':{'true_cnt':len(y), 'recall':recall, 'pre_cnt':len(y), 'prec':precision}}
        for label, group in groupby(y_comb, lambda x:x[0]):
            group = list(group)
            g_y = [x[0] for x in group]
            g_pre = [x[1] for x in group]
            label_recall = len([x for x in g_pre if x==label])/float(len(group))
            # label_recall = recall_score(g_y, g_pre, average='macro')
            label_results[label]= {'true_cnt':len(group), 'recall':label_recall, 'pre_cnt':0, 'prec':0.0}
            #logger.get().info('label: %s, cnt: %s, recall: %s', label, len(group), label_recall)
        # stat label prec
        y_comb.sort(key=lambda x:x[1])
        for label, group in groupby(y_comb, lambda x:x[1]):
            group = list(group)
            g_y = [x[0] for x in group]
            g_pre = [x[1] for x in group]
            label_prec = len([x for x in g_y if x==label])/float(len(group))
            # label_prec = precision_score(g_y, g_pre, average='macro')
            label_results[label]['pre_cnt'] = len(group)
            label_results[label]['prec'] = label_prec
        print '-'*80
        print 'Params=%s, recall=%f, Prec=%f' % (str(param_dict), recall, precision)
        print '\n'.join(['%s,%s,%s,%s,%s'%(k,v['true_cnt'],v['recall'],v['pre_cnt'],v['prec']) for k,v in label_results.items()])
        return label_results


def load_train_data(data_fpath):
    keys = ['label', 'text']
    with codecs.open(data_fpath, 'r', 'utf-8') as f:
        infos = [x.strip('\n').split('\t') for x in f]
        infos = [x for x in infos if x[0] and x[1]]
        infos = [{keys[i]:x[i] for i in range(len(keys))} for x in infos]
    print 'load %s test data form %s' % (len(infos), data_fpath)
    return infos

def train_test_split_label(data, test_size=0.2):
    data.sort(key=lambda x:x['label'])
    train_data = []
    test_data = []
    for label, group in groupby(data, lambda x:x['label']):
        group = list(group)
        test_index = random.sample(range(len(group)), int(test_size*len(group)))
        train_data += [x for i,x in enumerate(group) if i not in test_index]
        test_data += [x for i,x in enumerate(group) if i in test_index]
    train_data.sort(key=lambda x:x['label'])
    test_data.sort(key=lambda x:x['label'])
    x_train = [x['text'] for x in train_data]
    y_train = [x['label'] for x in train_data]
    x_test = [x['text'] for x in test_data]
    y_test = [x['label'] for x in test_data]
    return x_train, y_train, x_test, y_test


def cross_validation_label(clf, data, test_size=0.1, cv=10):
    results = []
    for i in range(cv):
        x_train, y_train, x_test, y_test = train_test_split_label(test_data, test_size)
        train_cnt = {'all':len(y_train)}
        for label, group in groupby(sorted(y_train), lambda x:x):
            group = list(group)
            # print label, len(group)
            train_cnt[label.replace('__label__','')] = len(group)
        clf.fit(x_train, y_train)
        label_result = clf.score_label(x_test, y_test)
        results.append(label_result)
    avg_result = {k:{'true_cnt':sum([x[k]['true_cnt'] for x in results])/len(results), 
                     'recall':sum([x[k]['recall'] for x in results])/len(results),
                     'pre_cnt':sum([x[k]['pre_cnt'] for x in results])/len(results),
                     'prec':sum([x[k]['prec'] for x in results])/len(results),
                    } for k,v in results[0].items()}
    print '-'*30, 'cv %s result'%(cv), '-'*30
    print '\n'.join(['%s\t%s\t%s\t%s\t%s\t%s'%(k,train_cnt[k],v['true_cnt'],v['recall'],v['pre_cnt'],v['prec']) for k,v in avg_result.items()])

def save_badcase(clf, data, data_fpath='../data/fasttext_badcase.txt'):
    results = []
    x_train, y_train, x_test, y_test = train_test_split_label(test_data, 0.2)
    clf.fit(x_train, y_train)
    pre_results = clf.predict_proba(x_test, k=3)
    label_results = [get_label_result(t,y.replace('__label__',''),p) for t,y,p in zip(x_test,y_test,pre_results)]
    badcases = [x for x in label_results if x['index'] > 1]
    save_label_results(data_fpath, badcases)

def get_label_result(text, y_true, pre_labels):
    rsp_labels = [x[0] for x in pre_labels]
    _index = rsp_labels.index(y_true) + 1 if y_true in rsp_labels else 201
    result = {
        'query' : text,
        'label' : y_true,
        'results' : pre_labels,
        'index' : _index,
    }
    return result

def save_label_results(data_fpath, results):
    with codecs.open(data_fpath, 'w', 'utf-8') as f:
        _ = map(lambda x:f.write('%s\t%s\t%d\t%s\n' % (
                                    x['query'],
                                    x['label'],
                                    x['index'],
                                    '\t'.join(['%s:%s'%(y[0],y[1]) for y in x['results']]),
                                )
                ),
                results,
            )
    print 'save %s results to %s success!' % (len(results), data_fpath)
    
if __name__ == '__main__':
    #logger.start('../log/fasttext_test.log', __name__, 'DEBUG')

    
    print FasttextClassifier.fasttext_tokenize(u'洪德路906', char=False)
    print FasttextClassifier.fasttext_tokenize(u'洪德路906', char=True)

    