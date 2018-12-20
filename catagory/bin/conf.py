#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2018-04-11 16:16:08
# @Author  : AlexTang (1174779123@qq.com)
# @Link    : http://t1174779123.iteye.com
# @Description : 

# nlp config
nlp_cfg = {
    'stopword_fpath'       : '../conf/stopwords_usr.huishi',
    'jieba_dict_fpath'     : './conf/mombaby.keywords',
    'synonym_fpath'        : './conf/sim_words.huishi',
    'g_stop_words_cfg'       : set(),
    'user_stopwords_cfg'     : set(),
    'g_ud_words_cfg'         : set([u'phone_t', u'email_t', u'int_t', u'date_t', u'time_t', u'float_t']+['entity_%s'%i for i in range(1000)]),
    'g_synonyms_cfg'         : [],
    'field_keywords_cfg'     : set(),
}

# intent config
intent_cfg = {
    'music.play': 0,
    'OTHERS': 1,
    'navigation.navigation': 2,
    'phone_call.make_a_phone_call': 3,
    'music.pause': 4,
    'navigation.cancel_navigation': 5,
    'music.next': 6,
    'navigation.open': 7,
    'navigation.start_navigation': 8,
    'phone_call.cancel': 9,
    'music.prev': 10,
}

# server config
server_cfg = {
    'server_model_path': './use_models',
    'entity_file_fpath': '../conf/train_test_merge_entities_0712.csv'
}
