#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2018-04-27 14:56:26
# @Author  : AlexTang (1174779123@qq.com)
# @Link    : http://t1174779123.iteye.com
# @Description : 


from __future__ import division
import os
import codecs
import math
import random
from itertools import groupby
from fasttext_util import FasttextClassifier
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
from intent_classify import IntentClassify
import pickle
import numpy as np
from log import g_log_inst as logger

class ModelComparer(object):

    @classmethod
    def compare(cls, model_a, model_b, test_data, data_fpath='../data/compare_results/compare_results.txt'):
        results = []
        for data in test_data:
            text = data['text']
            y_true = data['label']
            pre_a, proba_a = model_a.predict_proba(text, k=1)
            pre_b, proba_b = model_b.predict_proba(text, k=1)
            # 
            results.append({
                'text': text,
                'y_true': y_true.replace('__label__',''),
                'pre_a': pre_a.replace('__label__',''),
                'pre_b': pre_b.replace('__label__',''),
            })
        cls.stat_differents(results, data_fpath)
        x = [x['text'] for x in results]
        y = [x['y_true'] for x in results]
        pre_a = [x['pre_a'] for x in results]
        pre_b = [x['pre_b'] for x in results]
        result_a = model_a.stat_label(x, y, pre_a)
        result_b = model_b.stat_label(x, y, pre_b)
        return results, result_a, result_b

    @classmethod
    def stat_differents(cls, results, data_fpath):
        a_better = [x for x in results if x['pre_a']==x['y_true'] and x['pre_b']!=x['y_true']]
        b_better = [x for x in results if x['pre_b']==x['y_true'] and x['pre_a']!=x['y_true']]
        both_bad = [x for x in results if x['pre_b']!=x['y_true'] and x['pre_a']!=x['y_true']]
        a_better.sort(key=lambda x:x['y_true'])
        b_better.sort(key=lambda x:x['y_true'])
        both_bad.sort(key=lambda x:x['y_true'])
        # print results
        print '%s case a better, %s case b_better, %s both bad' % (len(a_better), len(b_better), len(both_bad))
        with codecs.open(data_fpath, 'w', 'utf-8') as f:
            f.write('a better : %s\n' % (len(a_better)))
            _ = map(lambda x:f.write('%s\t%s\t%s\t%s\n'%(
                x['text'],x['y_true'],x['pre_a'],x['pre_b'])),a_better)
            f.write('\nb better : %s\n' % (len(b_better)))
            _ = map(lambda x:f.write('%s\t%s\t%s\t%s\n'%(
                x['text'],x['y_true'],x['pre_a'],x['pre_b'])),b_better)
            f.write('\nboth_bad : %s\n' % (len(both_bad)))
            _ = map(lambda x:f.write('%s\t%s\t%s\t%s\n'%(
                x['text'],x['y_true'],x['pre_a'],x['pre_b'])),both_bad)
        print 'save stat results to %s success' % (data_fpath)

    @classmethod
    def get_predict_result(cls, model, test_data):
        pre_results = []
        for data in test_data:
            text = data['text']
            y_true = data['label']
            if y_true.encode('utf-8') == '无意图': 
                y_true = '0'
            else:
                y_true = '1'
            pred_l, proba = model.predict_proba(data, k=1)
            # print(pred_l, proba)
            pre_results.append({
                'text': text,
                'text_pred': model.token_func(data),
                'y_true': y_true.replace('__label__', ''),
                'pred_l': pred_l.replace('__label__', ''),
                'proba':proba,
            })
    @classmethod
    def get_predict_result_muti(cls, model, test_data):
        pre_results = []
        total_y = [test_data[i]['label'] for i in range(len(test_data))]
        set_ys = list(set(total_y))
        for data in test_data:
            text = data['text']
            y_true = data['label']
            for j in range(len(set_ys)):
                if set_ys[j] == y_true:
                    y_true = str(j)
                    break
            pred_l, proba = model.predict_proba(data,k=1)
            pre_results.append({
                'text': text,
                'text_pred': model.token_func(data),
                'y_true': y_true.replace('__label__', ''),
                'pred_l': pred_l.replace('__label__', ''),
                'proba':proba,
            })        

        return pre_results


def cross_validation_compare(model_a, model_b, test_data, cv=10, seed=27):
    data_list = train_test_split_average(test_data, cv, seed)
    results_a = []
    results_b = []
    pred_results = []
    for i in range(cv):
        test_data = data_list[i]
        train_data = reduce(lambda x,y:x.extend(y) or x, data_list[:i]+data_list[i+1:], [])
        random.shuffle(train_data)
        train_cnt = stat_label_cnt(train_data)

        x_train_a = [model_a.token_func(x['text']) for x in train_data]
        x_train_b = [model_b.token_func(x['text']) for x in train_data]
        y_train = [x['label'] for x in train_data]
        model_a.fit(x_train_a, y_train)
        model_b.fit(x_train_b, y_train)

        _results, result_a, result_b = ModelComparer.compare(
            model_a, model_b, test_data, '../data/compare_results/compare_results%s.txt'%(i))
        # prase train count
        for k in result_a.keys():
            result_a[k]['train_cnt'] = train_cnt[k] if k in train_cnt else 0
            result_b[k]['train_cnt'] = train_cnt[k] if k in train_cnt else 0
        print '-'*80
        print 'a:'
        print '\n'.join(['%s\t%s\t%s\t%s\t%s\t%s'%(
                k,v['train_cnt'],v['true_cnt'],v['recall'],v['pre_cnt'],v['prec'])
            for k,v in result_a.items()])
        print 'b:'
        print '\n'.join(['%s\t%s\t%s\t%s\t%s\t%s'%(
                k,v['train_cnt'],v['true_cnt'],v['recall'],v['pre_cnt'],v['prec'])
            for k,v in result_b.items()])
        results_a.append(result_a)
        results_b.append(result_b)
        pred_results.extend(_results)
    ModelComparer.stat_differents(pred_results,'../data/compare_results/compare_results.txt')
    avg_result_a = {
        label:{
            k:sum([x[label][k] for x in results_a if label in x])/len(results_a)
            for k in results_a[0]['all']
        }
        for label in results_a[0]
    }
    avg_result_b = {
        label:{
            k:sum([x[label][k] for x in results_b if label in x])/len(results_b)
            for k in results_b[0]['all']
        }
        for label in results_b[0]
    }
    print '-'*30, 'cv %s result'%(cv), '-'*30
    print 'a:'
    print '\n'.join(['%s\t%s\t%s\t%s\t%s\t%s'%(
            k,v['train_cnt'],v['true_cnt'],v['recall'],v['pre_cnt'],v['prec'])
        for k,v in avg_result_a.items()])
    print 'b:'
    print '\n'.join(['%s\t%s\t%s\t%s\t%s\t%s'%(
            k,v['train_cnt'],v['true_cnt'],v['recall'],v['pre_cnt'],v['prec'])
        for k,v in avg_result_b.items()])


def train_test_compare(model_a, model_b, train_a, train_b, test_data):
    random.shuffle(train_a)
    random.shuffle(train_b)
    train_cnt_a = stat_label_cnt(train_a)
    train_cnt_b = stat_label_cnt(train_b)
    x_train_a = [model_a.token_func(x['text']) for x in train_a]
    x_train_b = [model_b.token_func(x['text']) for x in train_b]
    print '|'.join(x_train_a[:10])
    print '|'.join(x_train_b[:10])
    y_train_a = [x['label'] for x in train_a]
    y_train_b = [x['label'] for x in train_b]
    model_a.fit(x_train_a, y_train_a)
    model_b.fit(x_train_b, y_train_b)

    _results, result_a, result_b = ModelComparer.compare(
        model_a, model_b, test_data, 
        '../data/compare_results/train_test_compare_results.txt')
    # prase train count
    for k in result_a.keys():
        result_a[k]['train_cnt'] = train_cnt_a[k]
        result_b[k]['train_cnt'] = train_cnt_b[k]
    print '-'*30, 'train_test_compare result', '-'*30
    print 'a:'
    print '\n'.join(['%s\t%s\t%s\t%s\t%s\t%s'%(
            k,v['train_cnt'],v['true_cnt'],v['recall'],v['pre_cnt'],v['prec'])
        for k,v in result_a.items()])
    print 'b:'
    print '\n'.join(['%s\t%s\t%s\t%s\t%s\t%s'%(
            k,v['train_cnt'],v['true_cnt'],v['recall'],v['pre_cnt'],v['prec'])
        for k,v in result_b.items()])
    
def get_total_fasttext_multi_sen_vec(model, train_data, test_data, total_data):
    random.shuffle(train_data)
    train_x = [model.token_func(x) for x in train_data]
    train_y = [x['label'] for x in train_data]
    test_x = [model.token_func(x) for x in test_data]
    test_y = [x['label'] for x in test_data]
    total_x = [model.token_func(x) for x in total_data]
    total_y = [x['label'] for x in total_data]
    y_total = []
    set_ys = list(set(total_y))
    label_dict = {}
    for i in range(len(set_ys)):
        label_dict[i] = set_ys[i]
    for i in range(len(total_y)):
        for j in range(len(set_ys)):
            if set_ys[j] == total_y[i]:
                y_total.append(str(j))
                break
                
    model.fit(total_x, y_total) 
    y_train = []
    #set_ys = list(set(train_y))
    for i in range(len(train_y)):
        for j in range(len(set_ys)):
            if set_ys[j] == train_y[i]:
                y_train.append(str(j))
                break
                
                
    y_test = []
    #set_ys = list(set(test_y))
    for i in range(len(test_y)):
        for j in range(len(set_ys)):
            if set_ys[j] == test_y[i]:
                y_test.append(str(j))
                break
    
    train_sen_cuts = []
    for i in range(len(train_x)):
        train_sen_cuts.append(train_x[i])
    test_sen_cuts = []
    for i in range(len(test_x)):
        test_sen_cuts.append(test_x[i])
        
    results = {'test_x': test_sen_cuts, 'test_y':y_test, 'train_x': train_sen_cuts, 'train_y': y_train}
    pickle.dump(results, open('../data/data/sentence_result.txt', 'w'))
    return label_dict
    
    
def get_total_fasttext_sen_vec(model, train_data, test_data, total_data):
    random.shuffle(train_data)
    train_x = [model.token_func(x) for x in train_data]
    train_y = [x['label'] for x in train_data]
    test_x = [model.token_func(x) for x in test_data]
    test_y = [x['label'] for x in test_data]
    total_x = [model.token_func(x) for x in total_data]
    total_y = [x['label'] for x in total_data]
    y_total = []
    set_ys = list(set(total_y))
    for i in range(len(total_y)):
        for j in range(len(set_ys)):
            if set_ys[j] == total_y[i]:
                if set_ys[j].encode('utf-8') == '无意图':
                    y_total.append('0')
                else:
                    y_total.append('1')
                break
    model.fit(total_x, y_total) 
    y_train = []
    set_ys = list(set(train_y))
    for i in range(len(train_y)):
        for j in range(len(set_ys)):
            if set_ys[j] == train_y[i]:
                if set_ys[j].encode('utf-8') == '无意图':
                    y_train.append('0')
                else:
                    y_train.append('1')
                break
                
                
    y_test = []
    set_ys = list(set(test_y))
    for i in range(len(test_y)):
        for j in range(len(set_ys)):
            if set_ys[j] == test_y[i]:
                if set_ys[j].encode('utf-8') == '无意图':
                    y_test.append('0')
                else:
                    y_test.append('1')
                break
    
    train_sen_cuts = []
    for i in range(len(train_x)):
        train_sen_cuts.append(train_x[i])
    test_sen_cuts = []
    for i in range(len(test_x)):
        test_sen_cuts.append(test_x[i])
        
    results = {'test_x': test_sen_cuts, 'test_y':y_test, 'train_x': train_sen_cuts, 'train_y': y_train}
    pickle.dump(results, open('../data/data/sentence_result.txt', 'w'))
    
def multi_train_test_validation(model, train_data, test_data, save_fpath='', silent=False):
    random.shuffle(train_data)
    train_x = [model.token_func(x) for x in train_data]
    train_y = [x['label'] for x in train_data]
    test_x = [model.token_func(x) for x in test_data]
    test_y = [x['label'] for x in test_data]
    y_train = []
    set_ys = list(set(train_y))
    label_dict = {}
    for i in range(len(set_ys)):
        label_dict[i] = set_ys[i]
    for i in range(len(train_y)):
        for j in range(len(set_ys)):
            if set_ys[j] == train_y[i]:
                y_train.append(str(j))
    model.fit(train_x, y_train) 
    _cases = ModelComparer.get_predict_result_muti(model, test_data)
    stat_results = model.stat_label(
        [x['text'] for x in _cases],
        [x['y_true'] for x in _cases],
        [x['pred_l'] for x in _cases]
    )
    _badcases = [x for x in _cases if x['y_true'] != x['pred_l']]
    # _badcases = [x for x in _cases if x['proba'] < 0.6 or x['y_true'] != x['pred_l']]
    _badcases.sort(key=lambda x:x['proba'], reverse=True)
    
    if save_fpath:
        save_predict_results(_badcases, save_fpath)
        # save_predict_results(_cases, save_fpath)

    print '-'*80
    print 'model_num=%s, recall=%f, Prec=%f, f1-score=%s' % (
        len(model.models), 
        stat_results['all']['recall'], 
        stat_results['all']['prec'],
        2*stat_results['all']['prec']*stat_results['all']['recall']/(stat_results['all']['recall']+stat_results['all']['prec']))
    if not silent:
        print '\n'.join(['%s\t%s\t%s\t%s\t%s\t%s'%(k,v['train_cnt'],v['true_cnt'],v['recall'],v['pre_cnt'],v['prec']) for k,v in stat_results.items()])
    return label_dict,stat_results, _cases

def binary_train_test_validation(model, train_data, test_data, save_fpath='', silent=False):
    random.shuffle(train_data)
    train_x = [model.token_func(x) for x in train_data]
    train_y = [x['label'] for x in train_data]
    test_x = [model.token_func(x) for x in test_data]
    test_y = [x['label'] for x in test_data]
    y_train = []
    set_ys = list(set(train_y))
    for i in range(len(train_y)):
        for j in range(len(set_ys)):
            if set_ys[j] == train_y[i]:
                if set_ys[j].encode('utf-8') == '无意图':
                    y_train.append('0')
                else:
                    y_train.append('1')
                break
    model.fit(train_x, y_train) 
    _cases = ModelComparer.get_predict_result(model, test_data)
    stat_results = model.stat_label(
        [x['text'] for x in _cases],
        [x['y_true'] for x in _cases],
        [x['pred_l'] for x in _cases]
    )
    _badcases = [x for x in _cases if x['y_true'] != x['pred_l']]
    # _badcases = [x for x in _cases if x['proba'] < 0.6 or x['y_true'] != x['pred_l']]
    _badcases.sort(key=lambda x:x['proba'], reverse=True)
    
    if save_fpath:
        save_predict_results(_badcases, save_fpath)
        # save_predict_results(_cases, save_fpath)

    print '-'*80
    print 'model_num=%s, recall=%f, Prec=%f, f1-score=%s' % (
        len(model.models), 
        stat_results['all']['recall'], 
        stat_results['all']['prec'],
        2*stat_results['all']['prec']*stat_results['all']['recall']/(stat_results['all']['recall']+stat_results['all']['prec']))
    if not silent:
        print '\n'.join(['%s\t%s\t%s\t%s\t%s\t%s'%(k,v['train_cnt'],v['true_cnt'],v['recall'],v['pre_cnt'],v['prec']) for k,v in stat_results.items()])
    return stat_results, _cases



def train_test_thrshold(model, train_data, test_data, thrs=[], silent=False):
    data_fpath='../data/train_test_results/tt_thrshold_results.txt'
    stat_results, cases = train_test_validation(model, train_data, test_data, save_fpath=data_fpath, silent=False)
    thr_results = stat_cases_thrshold(model, cases, thrs)
    return thr_results

def cross_validation_badcase(model, data, cv=5, seed=27,
    data_fpath='../data/badcases_proba.txt'):
    data_list = train_test_split_average(data, cv, seed)
    
    cases = []
    for i in range(cv):
        test_data = data_list[i]
        train_data = reduce(lambda x,y:x.extend(y) or x, data_list[:i]+data_list[i+1:], [])
        random.shuffle(train_data)
        _stat_result, _cases = train_test_validation(model, train_data, test_data)
        cases.extend(_cases)

    stat_results = model.stat_label(
        [x['text'] for x in cases],
        [x['y_true'] for x in cases],
        [x['pred_l'] for x in cases]
    )
    badcases = [x for x in cases if x['pred_l'] != x['y_true']]
    badcases = sorted(badcases, key=lambda x:x['proba'], reverse=True)
    save_predict_results(badcases, data_fpath)
    
    print '-'*30, 'cv %s result'%(cv), '-'*30
    print '\n'.join(['%s\t%s\t%s\t%s\t%s\t%s'%(
        k,
        v['train_cnt'] if 'train_cnt' in v else 0,
        v['true_cnt'] if 'true_cnt' in v else 0,
        v['recall'] if 'recall' in v else 0.0,
        v['pre_cnt'] if 'pre_cnt' in v else 0,
        v['prec'] if 'prec' in v else 0.0,) for k,v in stat_results.items()])
    return cases


def stat_cases_thrshold(model, cases, thrs=
    [0.05*x for x in range(18)]+[0.01*x+0.9 for x in range(10)]):
    def fix_label(case, thr):
        case['y_fix'] = case['pred_l'] if case['proba'] >= thr else u'无意图'
    thr_results = []
    for thr in thrs:
        _ = map(lambda x:fix_label(x,thr), cases)
        prec, recall = stat_prec_recall(cases)
        thr_results.append({'thr':thr, 'prec':prec, 'recall': recall})
        stat_results = model.stat_label(
            [x['text'] for x in cases],
            [x['y_true'] for x in cases],
            [x['y_fix'] for x in cases]
        )
        print '-'*30, 'thr %s result'%(thr), '-'*30
        print '\n'.join(['%s\t%s\t%s\t%s\t%s\t%s'%(
            k,
            v['train_cnt'] if 'train_cnt' in v else 0,
            v['true_cnt'] if 'true_cnt' in v else 0,
            v['recall'] if 'recall' in v else 0.0,
            v['pre_cnt'] if 'pre_cnt' in v else 0,
            v['prec'] if 'prec' in v else 0.0,) for k,v in stat_results.items()])
    
    for result in thr_results:
        print '%s\t%s\t%s' % (result['thr'], result['prec'], result['recall'])
    return thr_results


def stat_prec_recall(cases):
    true_pos = [x for x in cases if x['y_true'] != u'无意图']
    pred_pos = [x for x in cases if x['y_fix'] != u'无意图']
    right_pos = [x for x in true_pos if x['y_fix'] == x['y_true']]
    prec = len(right_pos)/len(pred_pos)
    recall = len(right_pos)/len(true_pos)
    return prec, recall


def cross_validation_thrshold(model, data, cv=5, seed=27, thrs=[0.7], 
    data_fpath='../data/cv_results/cv_thrshold_results.txt'):
    cases = cross_validation_badcase(model, data, cv, seed, data_fpath)
    def fix_label(case, thr):
        case['y_fix'] = case['pred_l'] if case['proba'] >= thr else u'无意图'
    for thr in thrs:
        _ = map(lambda x:fix_label(x,thr), cases)
        stat_results = model.stat_label(
            [x['text'] for x in cases],
            [x['y_true'] for x in cases],
            [x['y_fix'] for x in cases]
        )
        print '-'*30, 'thr %s result'%(thr), '-'*30
        print '\n'.join(['%s\t%s\t%s\t%s\t%s\t%s'%(
            k,
            v['train_cnt'] if 'train_cnt' in v else 0,
            v['true_cnt'] if 'true_cnt' in v else 0,
            v['recall'] if 'recall' in v else 0.0,
            v['pre_cnt'] if 'pre_cnt' in v else 0,
            v['prec'] if 'prec' in v else 0.0,) for k,v in stat_results.items()])
    


def stat_label_cnt(data):
    label_cnt = {'all':len(data)}
    for label, group in groupby(sorted([x['label'] for x in data]), lambda x:x):
        group = list(group)
        label_cnt[label.replace('__label__','')] = len(group)
    return label_cnt


def train_test_split_average(data, cv=5, seed=27):
    data_list = [[] for x in range(cv)]
    data.sort(key=lambda x:x['label'])
    for label, group in groupby(data, lambda x:x['label']):
        group = list(group)
        random.Random(seed).shuffle(group)
        sub_groups = chunks(group, cv)
        _ = map(lambda i:data_list[i].extend(sub_groups[i]), range(cv))
    print 'split dataset success: %s ' % ([len(x) for x in data_list])
    return data_list
#split the arr into N chunks
def chunks(arr, m):
    n = int(math.ceil(len(arr) / float(m)))
    _chunks = [arr[i:i + n] for i in range(0, len(arr), n)]
    while len(_chunks) < m:
        _chunks.append([])
    return _chunks


def save_predict_results(results, data_fpath):
    data_path = os.path.dirname(data_fpath)
    if not os.path.exists(data_path):
        os.mkdir(data_path)
    with codecs.open(data_fpath, 'w', 'utf-8') as f:
        f.write('text\ttrue_label\tpredict_label\tproba\n')
        _ = map(lambda x:f.write('%s\t%s\t%s\t%s\t%s\n'%(
            x['text'],x['text_pred'],x['y_true'],x['pred_l'],x['proba'])),results)
    print 'save %s badcases to %s ' % (len(results), data_fpath)


def load_train_data(data_fpath):
    keys = ['label', 'text']
    with codecs.open(data_fpath, 'r', 'utf-8') as f:
        infos = [x.strip('\n').split('\t') for x in f]
        infos = [x for x in infos if len(x)==2 and x[0] and x[1]]
        infos = [{keys[i]:x[i] for i in range(len(keys))} for x in infos]
    print 'load %s test data form %s' % (len(infos), data_fpath)
    return infos


if __name__ == '__main__':

    logger.start('../log/model_comparer.log', __name__, 'DEBUG')

    '''
        model compare code, fit different train data everytime
    '''
    
    # train_data = load_train_data('../data/kangmei_data/kangmei_confirm.txt')
    # train_data = load_train_data('../data/zhongtong_data/zhongtong_fasttext.txt')
    # train_data = load_train_data('../data/kangmei_data/kangmei_data_new.txt')
    train_data = load_train_data('../data/laiye_gen_sample_train.txt')
    # train_data = load_train_data('../data/origin_train.txt')
    test_data = load_train_data('../data/origin_develop.txt')
    # cur_model = IntentClassify([(2, 15, 150, 0.6)], use_rule=False)
    # cur_model = IntentClassify([(2, 10, 80, 0.2)], use_rule=False, 
    #     token_func=lambda x:FasttextClassifier.fasttext_tokenize(x, char=False),
    #     pretrain='../conf/w2v_sgns_win1_d80.kv')
    # cur_model = IntentClassify('../model/zhongtong/', use_rule=False, 
    #     token_func=lambda x:FasttextClassifier.fasttext_tokenize(x, char=False),
    #     pretrain='../conf/w2v_sgns_win1_d80.kv')

    # cur_model.train('../data/zhongtong_data/zhongtong_fasttext.txt', 
    #     '../model/zhongtong/')
    # cross_validation_badcase(cur_model, train_data, 5, '../data/kangmei_badcases.txt')
    # train_test_validation(cur_model, train_data, test_data, '../data/dev_badcase.txt')

    # train_data = load_train_data('../data/zhongtong_data/zhongtong_chat.txt')
    # test_data = load_train_data('../data/zhongtong_data/zto_messages.txt')
    # train_data = load_train_data('../data/kangmei_data/kangmei_data_0609.txt')
    # train_data = load_train_data('../data/kangmei_data/kangmei_comfirm_0609.txt')

    model_pretrain = IntentClassify(
        [(1, 10, 80, 0.2)], use_rule=False, 
        token_func=lambda x:FasttextClassifier.fasttext_tokenize(x, char=False),
        pretrain='../conf/w2v_sgns_win1_d80.kv')
    model_notrain = IntentClassify(
        [(1, 10, 80, 0.2)], use_rule=False, 
        token_func=lambda x:FasttextClassifier.fasttext_tokenize(x, char=False),
        pretrain='')
    model_pretrain_char_v4 = IntentClassify(
        [(1, 10, 100, 0.2)], use_rule=False, 
        token_func=lambda x:FasttextClassifier.fasttext_tokenize(x, char=True),
        pretrain='../conf/fasttext.model.min30_ngram5_d100_it20.vec')
    model_pretrain_char_v5 = IntentClassify(
        [(2, 10, 100, 0.2)], use_rule=False, 
        token_func=lambda x:FasttextClassifier.fasttext_tokenize(x, char=True),
        pretrain='../conf/fasttext.model.min30_ngram5_d100_it20.vec')
    model_notrain_char = IntentClassify(
        [(1, 10, 100, 0.2)], use_rule=False, 
        token_func=lambda x:FasttextClassifier.fasttext_tokenize(x, char=True),
        pretrain='')
    model_quantize = IntentClassify(
        [(2, 10, 100, 0.2)], use_rule=False, 
        token_func=lambda x:FasttextClassifier.fasttext_tokenize(x, char=True),
        pretrain='../conf/fasttext.model.min30_ngram5_d100_it20.vec',
        quantize=True)

    # cross_validation_badcase(model_notrain, train_data, 5, '../data/kangmei_badcases.txt')
    # cross_validation_badcase(model_pretrain, train_data, 5, '../data/kangmei_badcases.txt')
    # cross_validation_badcase(model_notrain_char, train_data, 5, '../data/kangmei_badcases.txt')
    # cross_validation_badcase(model_pretrain_char_v4, train_data, 5, '../data/kangmei_badcases.txt')
    cross_validation_badcase(model_pretrain_char_v5, train_data, 5, '../data/kangmei_badcases.txt')
    # cross_validation_compare(model_pretrain, model_notrain, train_data, 5)
    # cross_validation_compare(model_pretrain_char_v5, model_quantize, train_data, 5)
    # train_test_validation(model_pretrain_char_v5, train_data, test_data, 
    #     '../data/zhongtong_data/zto_tesult.txt')

    '''
        train test compare
    '''
    # train_a = load_train_data('../data/origin_train.txt')
    # train_b = load_train_data('../data/laiye_gen_sample_train.txt')
    # test_data = load_train_data('../data/origin_develop.txt')
    # test_data = load_train_data('../data/origin_test2017.txt')

    # ori_model = IntentClassify([(2, 15, 150, 0.6)], use_rule=False)
    # gene_model = IntentClassify([(2, 15, 150, 0.6)], use_rule=False)
    # train_test_compare(ori_model, gene_model, train_a, train_b, test_data)

    '''
        pretrain & char
    '''
    # pretrain_char = IntentClassify(
    #     [(2, 15, 100, 0.6)],
    #     use_rule=False, 
    #     token_func=lambda x:FasttextClassifier.fasttext_tokenize(x, char=True),
    #     pretrain='../conf/fasttext.model.min30_ngram5_d100_it20.vec')
    # pretrain_word = IntentClassify(
    #     [(2, 15, 80, 0.6)],
    #     use_rule=False, 
    #     token_func=lambda x:FasttextClassifier.fasttext_tokenize(x, char=False),
    #     pretrain='../conf/w2v_sgns_win1_d80.kv')
    # nontrain_char = IntentClassify(
    #     [(2, 15, 150, 0.6)],
    #     use_rule=False,
    #     token_func=lambda x:FasttextClassifier.fasttext_tokenize(x, char=True))
    # nontrain_word = IntentClassify(
    #     [(2, 15, 150, 0.6)],
    #     use_rule=False,
    #     token_func=lambda x:FasttextClassifier.fasttext_tokenize(x, char=False))
    
    # train_test_validation(pretrain_char, train_data, test_data, 
    #     save_fpath='../data/train_test_results/pretrain_char.txt')
    # train_test_validation(pretrain_word, train_data, test_data, 
    #     save_fpath='../data/train_test_results/pretrain_word.txt')
    # train_test_validation(nontrain_char, train_data, test_data, 
    #     save_fpath='../data/train_test_results/nontrain_char.txt')
    # train_test_validation(nontrain_word, train_data, test_data, 
    #     save_fpath='../data/train_test_results/nontrain_word.txt')
    # train_test_compare(pretrain_word, nontrain_char, train_data, train_data, test_data)
    

    # compare single & mutil model
    # single_model = IntentClassify([(2, 10, 100, 0.4)], use_rule=True)
    # intent_classify = IntentClassify(
    #     [(2, 10, 100, 0.4),(2, 10, 150, 0.4),(4, 10, 150, 0.4),(3, 10, 150, 0.4),(3, 10, 200, 1.0)],
    #     use_rule=True)
    # cross_validation_compare(single_model, intent_classify, train_data, 5)

    # compare token & char input
    # char_model = IntentClassify(
    #     [(2, 10, 100, 0.4),(2, 10, 150, 0.4),(4, 10, 150, 0.4),(3, 10, 150, 0.4),(3, 10, 200, 1.0)],
    #     use_rule=True)
    # token_model = IntentClassify(
    #     [(2, 10, 100, 0.4),(2, 10, 150, 0.4),(4, 10, 150, 0.4),(3, 10, 150, 0.4),(3, 10, 200, 1.0)],
    #     use_rule=True)
    # cross_validation_compare(char_model, token_model, train_data, 5)

    # compare params
    # model_a = IntentClassify(
    #     [(2, 10, 100, 0.4),(2, 10, 150, 0.4),(4, 10, 150, 0.4),(3, 10, 150, 0.4),(3, 10, 200, 1.0)],
    #     use_rule=True)
    # model_b = IntentClassify(
    #     [(3, 10, 100, 0.2),(4, 10, 100, 0.2),(3, 10, 200, 0.2),(3, 10, 150, 0.2),(3, 10, 200, 0.2)],
    #     use_rule=True)
    # cross_validation_compare(model_a, model_b, train_data, 5)
    
    # train_data = load_train_data('../data/NLPCC_fasttext_origin.txt')
    # train_test_split_average(train_data)
    # with_rule = IntentClassify(
    #     '../model/',use_rule=True)
    # without_rule = IntentClassify(
    #     '../model/',use_rule=False)
    # ModelComparer.compare(with_rule, without_rule, 
    #     [x['text'] for x in train_data], [x['label'] for x in train_data])
    # cross_validation_badcase(with_rule, train_data, 5, '../data/badcases_proba_sp.txt')

    '''
        predict some case 
    '''
    # print cur_model.predict(u'洪德路906')
    # print cur_model.predict(u'你能打电话吗')
    # print u'打开电话',cur_model.predict_proba(u'打开电话')
    # print u'打开酷狗音乐',cur_model.predict_proba(u'打开酷狗音乐')
    # print u'打个电话吧不打了',cur_model.predict_proba(u'打个电话吧不打了')
    # print u'融化了',cur_model.predict_proba(u'融化了')
    # print u'花匠',cur_model.predict_proba(u'花匠')
    # print cur_model.predict_proba(u'打开电话')
    # print cur_model.predict_proba(u'打电话')
    # print cur_model.predict(u'打电话', prev_infos=[{'text':'', 'intent':'music.play'}])

    # single_model.fit(
    #     [FasttextClassifier.fasttext_tokenize(x['text']) for x in train_data],
    #     [x['label'] for x in train_data])
    # print single_model.predict_proba(u'洪德路906')
    # print single_model.predict_proba(u'打电话')
