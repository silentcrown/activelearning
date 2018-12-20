#!/bin/env python3
#-*- encoding: utf-8 -*-

import os
import re
# import jieba_fast as jieba
import jieba
import codecs
import time
import itertools
from collections import Counter
from conf import nlp_cfg
#from log import g_log_inst as logger
from gensim import models
from gensim import corpora
from gensim import matutils


class NLPUtil(object):
    _max_kw_num = 10
    _valid_token_len = 5

    _wordseg_pattern_cfg = [
        re.compile(r'{.*?}', re.U),
    ]

    # _emoji_pattern_cfg = re.compile(r'[\U0001f600-\U0001f9ef]', re.U)
    _emoji_pattern_cfg = re.compile(u'('
        u'\ud83c[\udf00-\udfff]|'
        u'\ud83d[\udc00-\ude4f\ude80-\udeff]|'
        u'[\u2600-\u2B55])+', flags=re.UNICODE)

    _replace_pattern_cfg = {
        'phone_t'    : re.compile(r'1[0-9\*]{10}|\+861[0-9]{10}|[0-9]{3}-[0-9]{3}-[0-9]{4}|[0-9]{4}-[0-9]{7,8}|[8|6][0-9]{7}'),
        'email_t'    : re.compile(r'[^@|\s]+@[^@]+\.[^@|\s]+'),
        'float_t'    : re.compile(r'\d{1,2}\.\d{1,2}'),
        'date_t'     : re.compile(ur'\d{2,4}-\d{1,2}-\d{1,2}|\d{2,4}\.\d{1,2}\.\d{1,2}|\d{2,4}年\d{1,2}月\d{1,2}[号日]?|\d{2,4}年\d{1,2}月|\d{2,4}年|\d{1,2}月\d{1,2}[日号]|\d{1,2}月[份]?'),
        'time_t'     : re.compile(ur'\d{1,2}[:：]\d{1,2}[:：]\d{1,2}|\d{1,2}[:：]\d{1,2}|\d{1,2}点\d{1,2}分?|\d{1,2}点半?'),
        # 'ordernum_t' : re.compile(ur'[\w\*]{17,20}'),
    }

    replace_patterns = [
        ('phone_t'    , re.compile(r'1[0-9\*]{10}|\+861[0-9]{10}|[0-9]{3}-[0-9]{3}-[0-9]{4}|[0-9]{4}-[0-9]{7,8}|[8|6][0-9]{7}')),
        # ('ordernum_t' , re.compile(ur'[\w\*]{17,20}')),
        ('float_t'    , re.compile(r'\d{1,2}\.\d{1,2}')),
        ('email_t'    , re.compile(r'[^@|\s]+@[^@]+\.[^@|\s]+')),
        ('date_t'     , re.compile(ur'\d{2,4}-\d{1,2}-\d{1,2}|\d{2,4}\.\d{1,2}\.\d{1,2}|\d{2,4}年\d{1,2}月\d{1,2}[号日]?|\d{2,4}年\d{1,2}月|\d{2,4}年|\d{1,2}月\d{1,2}[日号]')),
        ('time_t'     , re.compile(ur'\d{1,2} :\d{1,2}:\d{1,2}|\d{1,2}:\d{1,2}|\d{1,2}点\d{1,2}分?|\d{1,2}点半?')),
        # ('', re.compile(ur'【[^!【】?]*】|\[.[^!\[\]?]*\]')),
    ]

    _illegal_char_set = set([])
    

    # init jieba
    jieba.initialize()

    # load user-define dictionary
    if os.path.exists(nlp_cfg['jieba_dict_fpath']):
        with codecs.open(nlp_cfg['jieba_dict_fpath'], 'r', 'utf8') as in_f:
            _ = map(lambda x: nlp_cfg['g_ud_words_cfg'].add(x.strip('\n')), in_f.readlines())
    for w in nlp_cfg['g_ud_words_cfg']:
        jieba.add_word(w, freq = 1000000)
    print 'load %d user-define jieba dict success!' % (len(nlp_cfg['g_ud_words_cfg']))

    # load stopwords
    if os.path.exists(nlp_cfg['stopword_fpath']):
        with codecs.open(nlp_cfg['stopword_fpath'], 'r', 'utf-8') as in_f:
            words = map(lambda w: w.strip('\n'), in_f.readlines())
            _ = map(lambda x: nlp_cfg['g_stop_words_cfg'].add(x), words)
        print 'load %s stopwords success!' % (len(words))

    # load synonyms
    if os.path.exists(nlp_cfg['synonym_fpath']):
        with codecs.open(nlp_cfg['synonym_fpath'], 'r', 'utf-8') as f:
            _ = map(lambda line: nlp_cfg['g_synonyms_cfg'].append(
                line.strip('\n').split(',')), f.readlines())
        print 'load synonyms success!'

    # load vip keywords
    # update_vip_words()

    # load tfidf model
    # _dictionary = corpora.Dictionary.load(nlp_cfg['dict_fpath'])
    # _model = models.tfidfmodel.TfidfModel.load(nlp_cfg['model_fpath'])


    @classmethod
    def remove_illegal_gbk_char(cls, text_unicode):
        try:
            text_unicode.encode('gbk')
            return text_unicode
        except UnicodeEncodeError as e:
            illegal_ch = e.object[e.start : e.end]
            illegal_set = cls._illegal_char_set
            illegal_set.add(illegal_ch)
            # try to replace directly
            for ch in illegal_set:
                text_unicode = text_unicode.replace(ch, '')
            # remove recursively
            return cls.remove_illegal_gbk_char(text_unicode)

    @classmethod
    def remove_emoji_char(cls, text_unicode):
        res = cls._emoji_pattern_cfg.sub('', text_unicode)
        return res

    @classmethod
    def conv_fenc_u8_to_gbk(cls, in_fpath, out_fpath):
        try:
            with codecs.open(in_fpath, 'r', 'utf-8') as rfd, \
                codecs.open(out_fpath, 'w', 'gbk') as wfd:
                # read utf8, write gbk
                for line in rfd:
                    line = cls.remove_illegal_gbk_char(line)
                    wfd.write(line)
        except Exception as e:
            logger.get().warn('errmsg=%s' % (e))

    @classmethod
    def tokenize_via_jieba(cls, text, normalize=True, filter_stop_word=True):
        # remove emoji
        # text = cls.remove_emoji_char(text)
        for s, p in cls.replace_patterns:
            text = re.sub(p, s, text)
        # normalize text
        if normalize:
            text = cls._normalize_text(text)
            # print 'after normalize_text:',text
        tokens = jieba.lcut(text.lower())
        if normalize:
            tokens = map(cls._normalize_token, tokens)
        if filter_stop_word:
            return filter(lambda x: x not in nlp_cfg['g_stop_words_cfg'], tokens)
        else:
            return tokens

    @classmethod
    def stat_token_freq(cls, in_fpath, out_fpath):
        stop_words = nlp_cfg['g_stop_words_cfg']
        try:
            word_counter = Counter()
            with codecs.open(in_fpath, 'r', 'utf-8') as rfd:
                for line in rfd:
                    raw_str, word_seg = line.strip('\n').split('\t')
                    tokens = word_seg.split()
                    tokens = filter(lambda x: x not in stop_words, tokens) 
                    tokens = map(cls._normalize_token, tokens)
                    for t in tokens:
                        if ('{[' not in t) and len(t) <= cls._valid_token_len:
                            word_counter[t] += 1
                        else:
                            logger.get().warn('invalid token, token=%s' % (t))
                            # tokenize via jieba 
                            for n_t in jieba.cut(t):
                                word_counter[n_t] += 1
                                logger.get().debug('jieba cut, token=%s' % (n_t))
            # dump word_counter
            sorted_words = sorted(word_counter.keys(),
                key = lambda k: word_counter[k], reverse = True)
            with codecs.open(out_fpath, 'w', 'utf-8') as wfd:
                for word in sorted_words:
                    tmp = '%s\t%s\n' % (word, word_counter[word]) 
                    wfd.write(tmp)
        except Exception as e:
            logger.get().warn('errmsg=%s' % (e))

    @classmethod
    def _normalize_token(cls, token):
        token = token.lower()
        try:
            # 11 usually means phone number
            if len(token) != 11 and token.isdigit():
                token = 'int_t'
            for k, v in cls._replace_pattern_cfg.items():
                if v.match(token):
                    token = k
                    break
            if '{[' not in token:
                return token
            for item in cls._wordseg_pattern_cfg:
                token = item.sub('', token)
            return token
        except Exception as e:
            logger.get().warn('token=%s, errmsg=%s' % (token, e))
            return token

    @classmethod
    def _normalize_text(cls, text): 
        the_patterns = []
        num_pattern = None
        for i,(name, pattern) in enumerate(cls.replace_patterns):
            if pattern.search(text):
                the_patterns.append((pattern, name))
        if not the_patterns:
            return text
        else:
            replaced_str = text
            for pattern, name in the_patterns:
                replaced_str = re.sub(pattern, name, replaced_str)
            return replaced_str

    @classmethod
    def get_sorted_keywords(cls, query):
        words = cls.tokenize_via_jieba(query)
        # add key words first
        sorted_words = filter(lambda x: x in nlp_cfg['field_keywords_cfg'], words)
        # then add sorted nonkey words
        sorted_words += cls._sort_nonkey_words(words)
        # add other words at last
        for word in words:
            if word not in sorted_words:
                sorted_words.append(word)
        return sorted_words[:cls._max_kw_num]

    # extract keywords for input_sug
    @classmethod
    def get_sorted_keywords_user(cls, query):
        words = cls.tokenize_via_jieba_fil(query)
        # add key words first
        sorted_words = filter(lambda x: x in nlp_cfg['field_keywords_cfg'], words)
        # then add sorted nonkey words
        sorted_words += cls._sort_nonkey_words(words)
        # add other words at last
        for word in words:
            if word not in sorted_words:
                sorted_words.append(word)
        return sorted_words

    @classmethod
    def get_dialog_keywords_gradually(cls, context, max_turn=3):
        query_keywords = []
        for i, query in enumerate(reversed(context)):
            # if over turn num, top extract
            if i > 2 * (max_turn-1):
                break
            keywords = []
            if not i % 2: # extract keywords from user msg
                keywords = cls.get_sorted_keywords_user(query)
                if max_turn > 1 & len(context) > 1: # won't limit keyword size if only extract from one turn
                    keywords = keywords[:nlp_cfg['max_usr_kw_num']]
                if not i == 0:
                    keywords = keywords[:nlp_cfg['max_con_kw_num']]
            else: # extract keywords from csr msg
                keywords = cls.get_sorted_keywords(query)[:nlp_cfg['max_con_kw_num']]
            
            query_keywords.extend([x for x in keywords if x not in query_keywords])
            # stop extract if enough
            if len(query_keywords) >= nlp_cfg['max_query_kw_num']:
                break
        return query_keywords[:nlp_cfg['max_query_kw_num']]

    @classmethod
    def get_dialog_keywords_with_index(cls, context):
        query_keywords = []
        for i, query in enumerate(reversed(context)):
            keywords = []
            if not i % 2:
                keywords = cls.get_sorted_keywords_user(query)[:nlp_cfg['max_usr_kw_num']]
                if not i == 0:
                    keywords = keywords[:nlp_cfg['max_con_kw_num']]
            else:
                keywords = cls.get_sorted_keywords(query)[:nlp_cfg['max_con_kw_num']]
            query_keywords.extend([(i, x) for x in keywords if x not in query_keywords])
            
            if len(query_keywords) >= nlp_cfg['max_query_kw_num']:
                break
        return query_keywords[:nlp_cfg['max_query_kw_num']]

    @classmethod
    def _sort_nonkey_words(cls, words):
        nonkey_words = filter(lambda x: x not in nlp_cfg['field_keywords_cfg'], words)
        tfidf_words = cls._get_tfidf(cls._dictionary, cls._model, nonkey_words)
        sorted_words = sorted(tfidf_words, key=lambda x: x[1], reverse=True)
        return [x[0] for x in sorted_words]

    @classmethod
    def _get_tfidf(cls, dictionary, tfidf_model, token_list):
        bow = dictionary.doc2bow(token_list)
        tfidf = tfidf_model[bow]
        # apply l1-norm to tfidf value
        tfidf = matutils.unitvec(tfidf, norm = 'l1')
        # id2token
        tfidf = map(lambda x: (dictionary[x[0]], x[1]), tfidf)
        # logger.get().debug('bow=%s, tfidf=%s' % (bow, tfidf))
        return tfidf

if '__main__' == __name__:
    logger.start('./log/test.log', __name__, 'DEBUG')
    # jingli_data_fpath  = '../files-data/jingli.question.data'
    # token_result_fpath = '../files-data/jingli.question.token.result.txt'
    # print '|'.join(NLPUtil.tokenize_via_jieba(u'1月1日到6月30日'))
    print '|'.join(NLPUtil.tokenize_via_jieba(u'&lt;&#x2F;li&gt;'))
    # print '|'.join(NLPUtil.tokenize_via_jieba_fil(u'[微笑]【收到不支持的消息类型，暂无法显示】%%【收到不支持的消息类型，暂无法显示】宝宝大便是干硬如颗粒状吗？'))
    print '|'.join(NLPUtil.tokenize_via_jieba(u'宝宝大便是干硬如颗粒状吗？'))
    print '|'.join(NLPUtil.tokenize_via_jieba(u'请问这个什么时候能到？'))
    print '|'.join(NLPUtil.tokenize_via_jieba(u'您太客气了，宝宝的健康成长也是我们共同的祝愿.[抱拳]    后期有问题欢迎随时咨询哒！[握手][爱心] 我们一直都在'))
    print '|'.join(NLPUtil.tokenize_via_jieba(u'您太客气了，宝宝的健康成长也是我们共同的祝愿.【抱拳】    后期有问题欢迎随时咨询哒！【握手】【爱心】 我们一直都在'))
    print '|'.join(NLPUtil.tokenize_via_jieba(u'【收到不支持的消息类型，暂无法显示】%%大约休息多久比较合适？'))
    print '|'.join(NLPUtil.tokenize_via_jieba(u'嗯嗯，多谢亲||不客气，应该哒%%祝小宝宝茁壮成长，每天开开心心%%还有其他的问题能帮您吗？||【收到不支持的消息类型，暂无法显示】%%没有了'))
    print NLPUtil._normalize_text('133****5454')
    print NLPUtil._normalize_text(u'[微笑]【收到不支持的消息类型，暂无法显示】%%【收到不支持的消息类型，暂无法显示】')
    print NLPUtil._normalize_text(u'1月1日到6月30日')
    # print '|'.join(NLPUtil.get_sorted_keywords(u'我想问问为什么注册超级会员，在账单里扣了10元，然后又单独扣8元?'))
    # print '|'.join(NLPUtil.get_sorted_keywords(u'因为湿疹是怕湿怕热的，所以给孩子洗脸的次数不要太多，每天1-2次即可；||此消息为【自定义表情或？】'))
    
    in_fpath = '../../simhash_remove_duplicate_tool/data/SE_query.txt'
    out_fpath = '../../simhash_remove_duplicate_tool/data/SE_query_long.txt'
    with codecs.open(in_fpath, 'r', 'utf8') as f_in, \
        codecs.open(out_fpath, 'w', 'utf8') as f_out:
        infos = [{
            'query':x, 
            'token':'|'.join(NLPUtil.tokenize_via_jieba(x.strip('\n'),normalize=False))
            } for x in f_in]
        print 'load %s querys from %s' % (len(infos), in_fpath)
        infos.sort(key=lambda x:x['token'])
        infos = [g.next() for k,g in itertools.groupby(infos, lambda x:x['token'])]
        infos = [x for x in infos if x['token'].count('|')>1]
        print 'get %s querys after tokenize' % (len(infos))
        _ = map(lambda x:f_out.write(x['query']), infos)
        print 'save %s querys to %s' % (len(infos), out_fpath)
    
