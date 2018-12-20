#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2018-04-26 20:32:36
# @Author  : AlexTang (1174779123@qq.com)
# @Link    : http://t1174779123.iteye.com
# @Description : 

from __future__ import unicode_literals
import os
import re
import time
import codecs
import functools
#from log import g_log_inst as logger

def cost(func):
    @functools.wraps(func)
    def wrapper(*args, **kw):
        start_time = time.time()
        ret = func(*args, **kw)
        logger.get().debug('@%s took %.6f seconds' % (func.__name__, time.time()-start_time))
        print '@%s took %.6f seconds' % (func.__name__, time.time()-start_time)
        return ret
    return wrapper


# rule config: (pos_pattern, neg_pattern, intent)
rule_cfg = [
    # (re.compile(ur'(唱|来|放|播|播放)[一]?(首|曲)'), re.compile(ur'小品|电视剧'), 'music'),
    (re.compile(ur'翻译|英语怎么说'), ur'', 'translation'),
    (re.compile(ur'天气'), ur'', 'weather'),
    (re.compile(ur'到.*的汽车'), ur'', 'bus'),
    (re.compile(ur'(高铁|火车)?![东南西北]?站'), ur'', 'train'),
    (re.compile(ur'简讯|短信'), ur'', 'message'),
    (re.compile(ur'([上中下]午|[晚早]上)[的]日程'), ur'', 'schedule'),
    (re.compile(ur''), ur'', ''),

]


def rule_predict(text):

    dict_label = '' # dict_rule(text, pred_label)
    hand_label = hand_rule(text)
    if dict_label:
        return dict_label
    elif hand_label:
        return hand_label
    return ''

def dict_rule(text, pre_label='OTHERS'):
    # instruction-base rule
    if text in instr_dict:
        return instr_dict[text]
    # slot-base rules
    match_slots = [(k,v) for k,v in slot_dict.items() if k == text]
    # if match_slots:
    #     print '%s\t%s' % (text, '\t'.join(['%s:%s'%(k,v) for k, v in match_slots]))
    if len(match_slots) == 1 and match_slots[0][1] in {'singer','song'}:
        if match_slots[0][1] == 'singer' and 'phone_call' in pre_label:
            return 'OTHERS'
        return 'music.play'
    return ''

def hand_rule(text):
    # pass through rules-by-hand
    for pos_pattern, neg_pattern, intent in rule_cfg:
        if not intent:
            continue
        # if exist, negetive pattern mustn't match
        if neg_pattern and neg_pattern.search(text):
            continue
        if pos_pattern.search(text):
            # print '%s match %s' % (text, pos_pattern.pattern)
            short_intents = {'music.pause', 'phone_call.cancel', 'navigation.cancel_navigation', 
            'phone_call.make_a_phone_call'}
            # if len(text) > 10 and intent in short_intents:
            #     return 'OTHERS'
            return intent
    return ''

def load_data(data_fpath):
    keys = ['label', 'text']
    with codecs.open(data_fpath, 'r', 'utf-8') as f:
        infos = [x.strip('\n').split('\t') for x in f]
        infos = [{keys[i]:x[i] for i in range(len(keys))} for x in infos]
    print 'load %s origin infos from %s ' % (len(infos), data_fpath)
    return infos

'''
    pass through all rules
'''
@cost
def pass_through_rules(infos):
    results = []
    for info in infos:
        label = rule_predict(info['text'],)
        if not label:
            continue
        results.append({
            'text':info['text'], 
            'label':info['label'],
            'pred_label':label,}
        )
    good_case = [x for x in results if x['label']==x['pred_label']]
    bad_case = [x for x in results if x['label']!=x['pred_label']]
    print 'pass through all rules, get %s good cases, %s bad cases' % (len(good_case), len(bad_case))
    return good_case, bad_case

def save_results(results, data_fpath):
    with codecs.open(data_fpath, 'w', 'utf-8') as f:
        _ = map(lambda x:f.write('%s\t%s\t%s\n'%(
                    x['text'], x['label'], x['pred_label'])
                ),
                results
            )

if __name__ == '__main__':
    logger.start('../log/rule_util.log', __name__, 'DEBUG')

    '''
        test rule_predict
    '''
    print rule_predict(u'遭遇')
    print rule_predict(u'帮我播放下一曲')
    print rule_predict(u'上一曲')
    print u'洪德路906', rule_predict(u'洪德路906')
    print u'来一首水调歌头', rule_predict(u'来一首水调歌头')
    
    
    infos_train = load_data('../data/origin_train.txt')
    infos_test = load_data('../data/origin_test2017.txt')
    infos_dev = load_data('../data/origin_develop.txt')
    infos = infos_train+infos_test+infos_dev
    good_case, bad_case = pass_through_rules(infos)
    save_results(good_case, '../data/rule_good_case.txt')
    save_results(bad_case, '../data/rule_bad_case.txt')

    # for info in infos:
    #     if info['query'] in slot_dict and info['intent'] != 'music.play':
    #         print '%s\t%s\t%s\t%s' % (info['sess_id'], info['query'], slot_dict[info['query']],info['intent']) 




