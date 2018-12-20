#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2018-04-10 19:10:47
# @Author  : AlexTang (1174779123@qq.com)
# @Link    : http://t1174779123.iteye.com
# @Description : 

import os
import re
import sys
import codecs
from copy import copy
from conf import intent_cfg
from itertools import groupby
from nlp_util_py3 import NLPUtil
from random import shuffle, sample


def load_origin_infos(data_fpath):
    keys = ['intent_id', 'intent', 'querys']
    with codecs.open(data_fpath, 'r', 'utf-8') as f:
        # content = f.read().split('\r\n')
        infos = [x.strip('\r\n').split('\t') for x in f if len(x)>2]
        
        infos = [{keys[i]:x[i] for i in range(len(keys))} for x in infos]
    print 'load %s infos from %s' % (len(infos), data_fpath)
    return infos

def load_data(data_fpath):
    keys = ['label', 'text']
    with codecs.open(data_fpath, 'r', 'utf-8') as f:
        infos = [x.strip('\n').split('\t') for x in f]
        infos = [{keys[i]:x[i] for i in range(len(keys))} for x in infos]
    print 'load %s origin infos from %s ' % (len(infos), data_fpath)
    return infos

def save_data(infos, data_fpath):
    with codecs.open(data_fpath, 'w', 'utf-8') as f:
        _ = map(lambda x:f.write('%s\t%s\n'%(x['label'], x['text'])), infos)
    print 'save %s data to %s' % (len(infos), data_fpath)


'''
    down sample data
'''
def down_sample_data(infos, train_fpath, test_fpath, total_fpath):
    infos = sorted(infos, key=lambda x:x['label'])
    train_data = []
    test_data = []
    for label, group in groupby(infos, lambda x:x['label']):
        group = list(group)
        shuffle(group)
        '''
        start1, start2 = len(test_data),len(train_data)
        if cnt >= len(group):
            test_data.extend(group[:atleast]) 
            train_data.extend(group[atleast:])
        else:
            cnt = int(3.0 * cnt / 4)
            train_data.extend(group[:cnt])
            test_data.extend(group[cnt:])
        print(label)
        print(len(train_data) - start2)
        '''
        train_data.extend(group[:int(3.0 * len(group) / 4)]) 
        test_data.extend(group[int(3.0 * len(group) / 4):]) 
    shuffle(test_data)
    shuffle(train_data)
    total_data = train_data + test_data
    save_data(train_data, train_fpath)
    save_data(test_data, test_fpath)
    save_data(total_data, total_fpath)


'''
    prase last turn intent to info
'''
def prase_last_intent(infos):
    prev_infos = [{'sess_id':0, 'intent':'OTHERS', 'query':''}]
    for info in infos:
        if prev_infos[-1]['sess_id'] != info['sess_id']:
            prev_infos = [{'sess_id':info['sess_id'], 'intent':'OTHERS', 'query':''}]
        info['prev_infos'] = copy(prev_infos)
        if info['intent'] != 'OTHERS':
            prev_infos.append(info)
    print 'prase %s infos success!'% (len(infos))
    return infos


'''
    stat session intent
'''
def stat_diff_intent(infos):
    infos.sort(key=lambda x:x['sess_id'])
    diff_sessions = []
    sess_cnt = 0
    for sess_id, session in groupby(infos, lambda x:x['sess_id']):
        sess_cnt += 1
        last_intent = ''
        session = list(session)
        for info in session:
            cur_intent = info['intent'].split('.')[0]
            if last_intent and 'OTHERS' not in (last_intent, cur_intent) and not last_intent == cur_intent:
                diff_sessions.append(session)
                break
            last_intent = info['intent'].split('.')[0]
    print 'from %s session find %s diff session' % (sess_cnt, len(diff_sessions))
    return diff_sessions

def print_session(session, slience=False):
    session_str = ''.join(['%s\t%s\t%s\t%s\n'%(
                            x['sess_id'], x['query'], x['intent'], x['marked_query']    
                            ) for x in session])
    if not slience:
        print session_str
    return session_str

def save_diff_session(data_fpath, diff_sessions):
    with codecs.open(data_fpath, 'w','  utf-8') as f:
        _ = map(lambda x:f.write(print_session(x, True)+'\n'), diff_sessions)
    print 'save %s diff sessions to %s' % (len(diff_sessions), data_fpath)

'''
    stat intent distribution, by query & session
'''
def stat_intent_distribution_query(infos):
    infos.sort(key=lambda x:x['intent'])
    stat_results = []
    for intent, info_group in groupby(infos, lambda x:x['intent']):
        stat_results.append({'intent':intent, 'infos':list(info_group)})
    stat_results.sort(key=lambda x:len(x['infos']), reverse=True)
    print '-'*30, 'intent_distribution_query', '-'*30
    for r in stat_results:
        print '%s\t%s' % (r['intent'], len(r['infos']))
    return stat_results

def stat_intent_distribution_tokenquery(infos):
    infos.sort(key=lambda x:x['intent'])
    stat_results = []
    for intent, info_group in groupby(infos, lambda x:x['intent']):
        info_group = list(info_group)
        # token remove same first
        for info in info_group:
            info['token'] = '|'.join(NLPUtil.tokenize_via_jieba(info['query']))
        info_group.sort(key=lambda x:x['token']+x['prev_infos'][-1]['intent'])
        info_group = [group.next() for key,group in groupby(info_group, lambda x:x['token']+x['prev_infos'][-1]['intent']) if key]
        info_group = [x for x in info_group if x['token']]
        stat_results.append({'intent':intent, 'infos':list(info_group)})
    stat_results.sort(key=lambda x:len(x['infos']), reverse=True)
    print '-'*30, 'intent_distribution_query_token', '-'*30
    for r in stat_results:
        avg_len = sum([len(x['token'].replace('|','')) for x in r['infos']])/float(len(r['infos']))
        avg_token = sum([len(x['token'].split('|')) for x in r['infos']])/float(len(r['infos']))
        print '%s\t%s\t%.2f\t%.2f' % (r['intent'], len(r['infos']), avg_len, avg_token)
    return stat_results

def stat_intent_distribution_session(infos):
    infos.sort(key=lambda x:x['sess_id'])
    stat_results = {}
    sess_cnt = 0
    for sess_id, session in groupby(infos, lambda x:x['sess_id']):
        sess_cnt += 1
        session = list(session)
        intents = {x['intent'] for x in session}
        for i in intents:
            if not i in stat_results:
                stat_results[i] = [0]
            stat_results[i].append(session)
    print 'stat %s sessions success!' % (sess_cnt)
    stat_results = [{'intent':intent, 'sessions':sessions} for intent, sessions in stat_results.items()]
    stat_results.sort(key=lambda x:len(x['sessions']), reverse=True)
    print '-'*30, 'intent_distribution_session', '-'*30
    for r in stat_results:
        print '%s\t%s' % (r['intent'], len(r['sessions']))
    return stat_results

'''
    stat intent query avg_len
'''
def stat_intent_querylen(infos):
    infos.sort(key=lambda x:x['intent'])
    stat_results = []
    for intent, info_group in groupby(infos, lambda x:x['intent']):
        info_group = sorted(info_group, key=lambda x:len(x['query']), reverse=True)
        avg_len = sum([len(x['query']) for x in info_group])/float(len(info_group))
        stat_results.append({'intent':intent, 'avg_len':avg_len})
    stat_results.sort(key=lambda x:x['avg_len'], reverse=True)
    print '-'*30, 'intent_querylen', '-'*30
    for r in stat_results:
        print '%s\t%s' % (r['intent'], r['avg_len'])
    return stat_results

def print_info(info):
    info_str = '\t'.join(info.values())
    print info_str
    return info_str
'''
    sample mark quality
'''
# def sample_mark_q

'''
    get fasttext train data from ori-infos
'''
def get_fasttext_train_data(stat_results, data_fpath):
    infos = reduce(lambda x,y:x.extend(y['infos']) or x, stat_results, [])
    for info in infos:
        info['fasttext'] = fasttext_tokenize(info['query'])
        cancel_p = re.compile(ur'关|暂停|关掉|关了|关闭|停掉|结束|退出|停止|取消|放弃|算了|不用了|拜拜|闭嘴')
        if len(info['query']) <= 3 and not cancel_p.search(info['query']) and info['prev_infos'][-1]['intent'] == 'phone_call.make_a_phone_call':
            cand_querys = [x['query'] for x in info['prev_infos'] if x['intent'] == 'phone_call.make_a_phone_call']
            # info['query'] = '|'.join(cand_querys)+'|'+info['query']
            info['query'] = info['prev_infos'][-1]['query']+'|'+info['query']
            print '%s\t%s\t%s\t%s' % (info['sess_id'], info['intent'], info['prev_infos'][-1]['intent'], info['query'])
                
    # shuffle(infos)
    with codecs.open(data_fpath,'w', 'utf-8') as f:
        # _ = map(lambda x:f.write('%s\t%s\n'%(x['intent'], x['query'])), infos)
        # _ = map(lambda x:f.write('%s\t%s %s\n'%(x['intent'], x['fasttext'], intent2str(x['pre_intent']))), infos)
        # _ = map(lambda x:f.write('%s\t%s\t%s\t%s\t%s\n'%(x['sess_id'], x['intent'], x['prev_infos'][-1]['intent'], x['prev_infos'][-1]['query'], x['query'])), infos)
        _ = map(lambda x:f.write('%s\t%s\t%s\t%s\n'%(x['sess_id'], x['intent'], x['prev_infos'][-1]['intent'], x['query'])), infos)

    print 'save %s infos to %s' % (len(infos), data_fpath)
def save_fasttext_train_data(infos, data_fpath):
    for info in infos:
        info['fasttext'] = fasttext_tokenize(info['text'])
    with codecs.open(data_fpath, 'w', 'utf-8') as f:
        _ = map(lambda x:f.write('__label__%s\t%s\n'%(x['label'],x['fasttext'])), infos)
    print 'save %s train data to %s success!' % (len(infos), data_fpath)


def fasttext_tokenize(text):
    uc_tokens = [u'http_t', u'email_t', u'date_t', u'phone_t', u'int_t']
    token = NLPUtil.tokenize_via_jieba(text, normalize=True, filter_stop_word=True)
    char_str = ' '.join([x if x in uc_tokens else ' '.join(x) for x in token])
    # char_str = ''.join(token)
    # print char_str
    return char_str
'''
    intent to string
'''
def intent2str(intent):
    num = intent_cfg[intent]
    onehot_list = ['1' if i==num else '0' for i in range(len(intent_cfg))]
    onehot_str = ' '.join(onehot_list)
    return onehot_str


'''
    get LSTM train data from ori-infos
'''
def get_LSTM_train_data(stat_results, data_fpath):
    infos = reduce(lambda x,y:x.extend(y['infos']) or x, stat_results, [])
    for info in infos:
        info['token'] = '|'.join(NLPUtil.tokenize_via_jieba(info['query'], normalize=True, filter_stop_word=True))
    shuffle(infos)
    with codecs.open(data_fpath,'w', 'utf-8') as f:
        _ = map(lambda (i,x):f.write('%s\t%s\t%s\t%s\t%s\t%s\n'%(i, x['intent'], x['query'], 'none', 0, x['token'])), enumerate(infos))
    print 'save %s infos to %s' % (len(infos), data_fpath)

def train_test_vaild_split_label(data_fpath, test_size=0.1, vaild_size=0.1):
    with codecs.open(data_fpath, 'r', 'utf-8') as f:
        data = [x.strip('\n').split('\t') for x in f]
    print 'load %s data from %s' % (len(data), data_fpath)
    data.sort(key=lambda x:x[1])
    train_data = []
    test_data = []
    vaild_data = []
    for label, group in groupby(data, lambda x:x[1]):
        group = list(group)
        test_index = sample(range(len(group)), int(test_size*len(group)))
        temp_data = [x for i,x in enumerate(group) if i not in test_index]
        test_data += [x for i,x in enumerate(group) if i in test_index]
        vaild_index = sample(range(len(temp_data)), int(vaild_size*len(group)))
        train_data += [x for i,x in enumerate(temp_data) if i not in vaild_index]
        vaild_data += [x for i,x in enumerate(temp_data) if i in vaild_index]
    shuffle(train_data)
    shuffle(test_data)
    shuffle(vaild_data)
    print len(train_data), len(test_data), len(vaild_data)
    with codecs.open('../data/trainset.txt', 'w', 'utf-8') as f:
        _ = map(lambda x:f.write('\t'.join(x)+'\n'), train_data)
    with codecs.open('../data/testset.txt', 'w', 'utf-8') as f:
        _ = map(lambda x:f.write('\t'.join(x)+'\n'), test_data)
    with codecs.open('../data/validset.txt', 'w', 'utf-8') as f:
        _ = map(lambda x:f.write('\t'.join(x)+'\n'), vaild_data)
    
def load_slot_dict(data_path):
    slot_dict = {}
    for filename in os.listdir(data_path):
        data_fpath = os.path.join(data_path, filename)
        with codecs.open(data_fpath, 'r', 'utf-8') as f:
            slots = [x.strip('\r\n') for x in f]
            for slot in slots:
                slot_dict[slot] = filename.replace('.txt','')
            print 'load %s slots from %s' % (len(slots), filename)
    print 'load %s slots from %s ' % (len(slot_dict), data_path)
    return slot_dict

'''
    find special samples
'''
def find_short_phone_query(infos, data_fpath):
    cancel_p = re.compile(ur'关|暂停|关掉|关了|关闭|停掉|结束|退出|停止|取消|放弃|算了|不用了|拜拜|闭嘴')
    with codecs.open(data_fpath, 'w', 'utf-8') as f:
        for info in infos:
            if len(info['query']) <= 3 and not cancel_p.search(info['query']) and info['prev_infos'][-1]['intent'] == 'phone_call.make_a_phone_call':
                cand_querys = [x['query'] for x in info['prev_infos'] if x['intent'] == 'phone_call.make_a_phone_call']
                print '%s\t%s\t%s\t%s' % (info['sess_id'], info['intent'], info['prev_infos'][-1]['intent'], '|'.join(cand_querys)+'|'+info['query'])
                f.write('%s\t%s\t%s\t%s\n' % (
                    info['sess_id'], 
                    info['intent'], 
                    info['prev_infos'][-1]['intent'], 
                    '|'.join(cand_querys)+'|'+info['query']))

def load_train_data(data_fpath):
    keys = ['sess_id', 'label', 'pre_label', 'pre_text', 'text']
    with codecs.open(data_fpath, 'r', 'utf-8') as f:
        infos = [x.strip('\n').split('\t') for x in f]
        infos = [x for x in infos if x[0] and x[1]]
        infos = [{keys[i]:x[i] for i in range(len(keys))} for x in infos]
    print 'load %s test data form %s' % (len(infos), data_fpath)
    return infos

def save_zhongtong_data(infos, data_fpath):
    new_infos = []
    for info in infos:
        new_infos += [{
            'intent_id':info['intent_id'],
            'intent':info['intent'],
            'text':x
        } for x in info['querys'].split('$$')]
    print 'get %s data from %s infos' % (len(new_infos), len(infos))
    with codecs.open(data_fpath, 'w', 'utf-8') as f:
        _ = map(lambda x:f.write(
                    '%s\t%s\n'%(x['intent_id']+x['intent'],x['text'])),
                new_infos
            )
    print 'save %s data to %s' % (len(new_infos), data_fpath)
    return new_infos


def save_baole_origin(in_fpath, out_fpath):
    keys = ['type', 'title', 'knows']
    with codecs.open(in_fpath, 'r', 'utf-8') as f:
        infos = [x.strip('\n').split('\t') for x in f]
        infos = [{keys[i]:x[i] for i in range(len(keys))} for x in infos]
    data = [[{'label':x['type'], 'text':y} for y in [x['title']]+x['knows'].split('||')] for x in infos]
    data = reduce(lambda x,y:x+y, data, [])
    print 'load %s data from %s' %(len(data), in_fpath)
    with codecs.open(out_fpath, 'w', 'utf-8') as f:
        _ = map(lambda x:f.write('%s\t%s\n'%(x['label'],x['text'])), data)
    print 'save %s data to %s success!' % (len(data), out_fpath)

if __name__ == '__main__':
    # infos = load_origin_infos('../data/corpus.train.txt')
    # infos = load_origin_infos('../data/zhongtong_data/zhongtong_task.txt')
    # save_zhongtong_data(infos, '../data/zhongtong_data/zhongtong_fasttext.txt')

    # infos = load_data('../data/zhongtong_data/zhongtong_fasttext.txt')
    # down_sample_data(infos, 
    #     '../data/zhongtong_data/zhongtong_train.txt',
    #     '../data/zhongtong_data/zhongtong_test.txt',
    #     cnt=10)

    save_baole_origin('../data/baole_data/baole_after_sale.txt', '../data/baole_data/baole_fasttext.txt')

    