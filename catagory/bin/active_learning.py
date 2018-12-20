
# coding: utf-8

# In[1]:



import pandas as pd
import numpy as np
import random
from intent_classify import IntentClassify
from gensim import corpora, models, similarities,matutils
from fasttext_util import FasttextClassifier
import time
from collections import Counter
import re
from url import Searcher
import pickle
#import matplotlib.pyplot as plt
from sklearn import metrics
#get_ipython().magic(u'matplotlib inline')


# In[2]:
print 'start ...'

class word():    
    def __init__(self):
        self.word_to_vec = {}
        self.dictionary = corpora.dictionary.Dictionary()
        self.dic = self.dictionary.load('/home/works/yexin/yexin/catagory/util/new/laiye/corpus.dict')
        self.tfidf = models.TfidfModel.load('/home/works/yexin/yexin/catagory/util/new/laiye/corpus.tfidf_model')  
        
    def remove_punctuation(self, line):
        rule = re.compile(ur"[^a-zA-Z0-9\u4e00-\u9fa5]")
        line = rule.sub('',line)
        return line        

    def stopwordslist(self, filepath):  
        stopwords = [line.strip() for line in open(filepath, 'r').readlines()]  
        return stopwords  


    # 对句子进行分词  
    def jieba_cut(self, sentence):  
        sentence_seged = jieba.cut(sentence.strip())  
        stopwords = stopwordslist('/home/works/yexin/yexin/catagory/util/new/laiye/stopwords.txt')  # 这里加载停用词的路径  
        outstr = ''  
        for word in sentence_seged:  
            if word.encode('utf-8') not in stopwords:  
                if word.encode('utf-8') != '\t' and word != '\t':  
                    outstr += word
                    outstr += " "  
        return outstr.strip().split(' ')

    def get_word2vec(self):
        with open('/home/works/yexin/yexin/catagory/util/new/laiye/w2v_sgns_win1_d80.kv') as f:
            data = [x.split(' ') for x in f.readlines()[1:]]
            words = [d[0] for d in data]
            vecs = np.array([d[1: -1] for d in data], dtype= 'float64')
            for i in range(len(words)):
                word = words[i]
                self.word_to_vec[word] = vecs[i]

    def get_tf_idf_of_query(self, query, dic, tfidf):      
        vec_bow = dic.doc2bow(query)
        vec_tfidf = tfidf[vec_bow]
        tp = [0.0] * len(query)
        ids = [tid[0] for tid in vec_tfidf]
        flags = [0] * len(query)
        count = 0.00001
        for j in range(len(query)):
            for i in range(len(ids)):
                if self.dic[ids[i]] == query[j]:
                    tp[j] = vec_tfidf[i][1]
                    flags[j] = 1
                    count += 1
                    break
        sums = sum(tp)
        #print ','.join(query)
        for i in range(len(flags)):
            if flags[i] == 1:
                continue
            tp[i] = 1.0 * sums / count
                    
        # apply l1-norm to tfidf value
        #tfidf = matutils.unitvec(vec_tfidf, norm = 'l1')   
        return tp

    def get_vectors_of_data_cut(self, data):
        vecs = []
        for i in range(len(data)):
            vec = []
            d_line = data[i].split(' ')
            tfidfs = self.get_tf_idf_of_query(d_line, self.dic, self.tfidf)
            s = sum(tfidfs)
            for j in range(len(d_line)):
                if d_line[j].encode('utf-8') in self.word_to_vec:
                    if s == 0:
                        vec.append([0.0] * len(self.word_to_vec['家']))
                    else:
                        vec.append((tfidfs[j] / s) * self.word_to_vec[d_line[j].encode('utf-8')])
            if len(vec) == 0:
                vec.append([0.0] * len(self.word_to_vec['家']))
            vecs.append(np.sum(np.array(vec), axis = 0))
        return np.array(vecs)    
    


# In[24]:


class data():
    
    def __init__(self, data_path, text_path, knowledge_path, label_path):
        self.data = open(data_path, 'r').readlines()
        self.knowledge = open(knowledge_path, 'r').readlines()
        self.text_path = text_path
        self.label = open(label_path, 'r').readlines()
        self.indexes = []
        self.labeled_indexes = []
        self.unlabeled_indexes = []
        self.test_indexes = []
        self.questions_match_dict = []
        self.t_map = {}
        self.range = [0.000001,0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5,0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9,0.95,1.0]
        self.range_result = {0.05:0, 0.1:0,0.15:0, 0.2:0,0.25:0, 0.3:0,0.35:0, 0.4:0,0.45:0,0.5:0, 0.55:0,0.6:0,0.65:0,0.7:0, 0.75:0,0.8:0,0.85:0,0.9:0,0.95:0, 1.0:0}
        self.dic = {}
        
    def get_texts(self, text_path):
        texts = []
        knowledges = []
        labels = []
        for i in range(len(self.data)):
            for key in self.data[i].split('||'):
                if key.strip() != '':
                    key = key.strip().replace('\n','')
                    texts.append(key) 
                    knowledges.append(self.knowledge[i].strip().replace('\n',''))
                    labels.append(self.label[i].strip().replace('\n',''))
        text_csv = pd.DataFrame({'text':texts, 'knowledge':knowledges, 'label':labels})
        text_csv.to_csv(text_path, index = False, encoding = 'utf-8')
        
    def split_real_labeled_unlabeld_and_test(self):
        self.get_texts(self.text_path)
        data = pd.read_csv(self.text_path)
        undefined = pd.read_csv('/home/works/yexin/yexin/catagory/data/ooooo.csv')
        labeled, unlabeled = data, undefined
        samp_index = range(len(list(undefined['rq_question'])))
        random.shuffle(samp_index)
        samp_index = samp_index[:10000]
        remain = []
        res = []
        l = list(data['text'])
        for i in range(len(l)):
            res.append((l[i], i))
        l_u = list(undefined['rq_question'])
        for i in range(len(l_u)):
            if i in samp_index:
                res.append((l_u[i], i))
        pickle.dump(res, open('total_texts.txt', 'w'))
        print 'generating'
        return labeled, unlabeled
    
    def split_labeled_unlabeld_and_test(self):
        self.get_texts(self.text_path)
        data = pd.read_csv(self.text_path)
        l = list(data['text'])
        k = list(data['knowledge'])
        lab = list(data['label'])
        '''
        res = []
        for i in range(len(l)):
            res.append((l[i], i))
        pickle.dump(res, open('total_texts.txt', 'w'))
        
        import ipdb;
        ipdb.set_trace()
        '''
        unique_knowledge = data['knowledge'].unique()
        self.labeled_indexes = []
        #data = data[data['label'] != '聊天']
        for i in range(len(unique_knowledge)):
            data_of_know = data[data['knowledge'] == unique_knowledge[i]]
            inds = data_of_know.index.tolist()
            random.shuffle(inds)
            if len(inds) < 10:
                self.labeled_indexes = self.labeled_indexes + inds[:int(0.5*len(inds)) + 1]
            else:
                self.labeled_indexes = self.labeled_indexes + inds[:10]
            
        self.indexes = data.index.tolist()
        left_indexes = list(set(self.indexes) - set(self.labeled_indexes))
        random.shuffle(left_indexes)
        self.unlabeled_indexes, self.test_indexes = left_indexes[:int(0.8 * len(left_indexes))], left_indexes[int(0.8 * len(left_indexes)) :]
        labeled, unlabeled, test = data.loc[self.labeled_indexes], data.loc[self.unlabeled_indexes], data.loc[self.test_indexes]
        print len(self.unlabeled_indexes),len(self.test_indexes),len(self.labeled_indexes),len(self.indexes)
        test.to_csv('/home/works/yexin/yexin/catagory/data/test_undefined.csv')
        return labeled, unlabeled, test
    
    def trans_i_to_str_dic(self, dic):
        i_to_str_map = pickle.load(open('/home/works/yexin/yexin/catagory/total_texts.txt'))
        for i in range(len(i_to_str_map)):
            self.t_map[i_to_str_map[i][1]] = i_to_str_map[i][0]
        new_dic = {}
        for key in dic:
            new_dic[self.t_map[key]] = dic[key]
        return new_dic
    
    def get_dict_of_questions(self):
        self.dic = pickle.load(open('/home/works/yexin/yexin/catagory/data/total_dic_results_xiaolai.txt', 'r'))#pickle.load(open('../data/total_dic_results.txt', 'r'))
        '''
        for key in self.dic:
            sim_result = self.dic[key]
            scores = [sim_result[k][2] for k in range(len(sim_result))]
            for m in range(len(scores)):
                for j in range(len(self.range) - 1):
                    if self.range[j] < float(scores[m]) <= self.range[j + 1]:
                        self.range_result[self.range[j + 1]] += 1
        xl = [0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5,0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9,0.95,1.0]
        col_value = []
        for i in range(len(xl)):
            col_value.append(self.range_result[xl[i]])
        plt.bar(range(5,105,5), col_value)
        plt.show()
        '''

        
        def update_unlabeled_data(self, new_undefined_path):
            origin_texts = [key for key in self.dic]
            cur_undefined = pd.read_csv(new_undefined_path)
            cur_texts = list(new_undefined['text'])
            new_texts = list(set(cur_texts) - set(origin_texts))
            return new_texts
        
        return self.dic


# In[25]:


class active_learning():
    
    def __init__(self, labeled, unlabeled, test, dic,  k, word_util, map_):
        self.labeled = labeled
        self.unlabeled = unlabeled
        self.test = test
        self.labeled_texts = list(labeled['text'])
        self.unlabeled_texts = list(unlabeled['rq_question'])
        #self.test_texts = list(test['text'])
        self.dic = dic
        #self.real = list(test['knowledge'])
        self.unlabeled_indexes = unlabeled.index.tolist()
        self.k = k
        self.sets = []
        self.map = map_
        self.word_util = word_util
        
    def predict(self):
        pred = []
        print 'now labeled sets size is : ' + str(len(self.labeled_texts))
        for i in self.test.index.tolist():
            results = self.dic[i]
            res_in = []
            for res in results:
                ques, title, score = res
                if ques.encode('utf-8') in self.labeled_texts:
                    knowledge = self.labeled[self.labeled['text'] == ques.encode('utf-8')]['knowledge']
                    knowledge = knowledge.loc[knowledge.index.tolist()[0]]
                    res_in.append(knowledge)
            if len(res_in) == 0:
                res_in.append('聊天')
            sort_dic = sorted(Counter(res_in).items(), key = lambda item:item[1], reverse = True)
            final_pred = sort_dic[0][0]
            pred.append(final_pred)
        real = [self.real[i].decode('utf-8') for i in range(len(self.real))]
        preds = [pred[i].decode('utf-8') for i in range(len(pred))]
        print('classification_report :\n%s' % metrics.classification_report(real, preds)) 
        
    def select_set(self):
        ent_dict = {}
        count_dict = []
        for i in self.unlabeled_indexes[:len(self.unlabeled)]:
            results = dic[self.unlabeled.loc[i,'rq_question'].decode('utf-8')]
            res_in = []
            if i % 100 == 0:
                print i
            if results != None:
                for res in results:
                    ques, title, score = res
                    if ques.encode('utf-8') in self.labeled_texts and 0.45 <= float(score):
                        knowledge = self.labeled[self.labeled['text'] == ques.encode('utf-8')]['knowledge']
                        knowledge = knowledge.loc[knowledge.index.tolist()[0]]
                        res_in.append((knowledge, float(score)))
                if len(res_in) == 0:
                    res_in.append(('聊天',0.0000001))
                counter = {}
                total_score = 0.0000001
                for k in range(len(res_in)):
                    if res_in[k][0] not in counter:
                        counter[res_in[k][0]] = 0.0000001
                    counter[res_in[k][0]] += res_in[k][1]
                    total_score += res_in[k][1]
                ent = 0.0000001
                scores,ents, ent_dic = [], [], {}
                for key in counter:
                    scores.append(counter[key] / total_score)
                    p = counter[key] / total_score
                    logp = np.log2(p)
                    ent -= p * logp
            ents.append(ent)     
            ent_dict[i] = ent
            count_dict.append(counter)
        '''
        pickle.dump(ent_dict, open('../util/ent_dict.txt', 'w'))
        pickle.dump(count_dict, open('../util/count_dict.txt', 'w'))
        ent_dict = pickle.load(open('../util/ent_dict.txt', 'r'))
        count_dict = pickle.load(open('../util/count_dict.txt', 'r'))
        '''
        del_ = []
        for k_i in ent_dict:
            if type(self.unlabeled.loc[k_i,'rq_question']) != str or '提醒' in self.unlabeled.loc[k_i,'rq_question'] or self.unlabeled.loc[k_i,'choose'] == 0:
                del_.append(k_i)
        for d_i in del_:
            del ent_dict[d_i]
        sort_dic = sorted(ent_dict.items(), key = lambda item:item[1], reverse = True)[:self.k]
        cur_set_index = [sort_dic[k][0] for k in range(len(sort_dic))]
        cur_set_entropy = [sort_dic[k][1] for k in range(len(sort_dic))]
        cur_count = []
        for i in range(len(count_dict)):
            if i in cur_set_index:
                cur_count.append(count_dict[i])
        
        return cur_set_index, cur_set_entropy, cur_count
            
    def merge(self, cur_set, cur_set_entropy, cur_count):
        self.sets = [] 
        new_labeled = pd.concat((self.labeled, self.unlabeled.loc[cur_set]), axis = 0)
        print 'merge into file : '+ '/home/works/yexin/yexin/catagory/data/total_results/new_labeled_sum.csv'
        self.unlabeled.loc[cur_set,'rq_question'].to_csv('/home/works/yexin/yexin/catagory/data/total_results/new_labeled_sum.csv') 
        new_unlabeled = self.unlabeled.drop(cur_set, axis = 0)
        self.labeled = new_labeled
        self.unlabeled = new_unlabeled
        self.labeled_texts = list(self.labeled['text'])
        self.unlabeled_texts = list(self.unlabeled['rq_question'])
        self.unlabeled_indexes = new_unlabeled.index.tolist()        


# In[26]:


#if __name__=="__main__":
data_util = data('/home/works/yexin/yexin/catagory/data/data.txt', '/home/works/yexin/yexin/catagory/data/text.csv', '/home/works/yexin/yexin/catagory/data/knowledge.txt', '/home/works/yexin/yexin/catagory/data/label.txt')
word_util = word()
word_util.get_word2vec()
#labeled, unlabeled, test = data_util.split_labeled_unlabeld_and_test()
labeled, unlabeled = data_util.split_real_labeled_unlabeld_and_test()
test = None
labeled_texts = labeled['text']
#test_texts = test['text']
#dic = data_util.get_dict_of_questions()


# In[27]:

print 'loading dictionary ...'
#t_dic = data_util.get_dict_of_questions()
dic = data_util.get_dict_of_questions()
#dic = data_util.trans_i_to_str_dic(t_dic)

print 'length of dict is :' + str(len(dic))
# In[28]:


labeled_indexes, unlabeled_indexes = data_util.labeled_indexes, data_util.unlabeled_indexes
al = active_learning(labeled, unlabeled, test, dic, 500, word_util, data_util.t_map)
while len(al.unlabeled_indexes) > 500:
    print 'length of cur unlabeled index is : ' + str(len(al.unlabeled_indexes))
    #al.predict()
    cur_set, cur_set_entropy, cur_count = al.select_set()
    al.merge(cur_set, cur_set_entropy, cur_count)
    break


# In[45]:


pd.read_csv('/home/works/yexin/yexin/catagory/data/ooooo.csv').loc[2532]


# In[46]:


ddd = pd.read_csv('/home/works/yexin/yexin/catagory/data/ooooo.csv')


# In[47]:


new_ddd = pd.read_csv('/home/works/yexin/yexin/catagory/data/new_ooo.csv')
new_ddd


# In[51]:


ddd.loc[4048]


# In[55]:


ind = new_ddd[new_ddd['choose'] == 0].index.tolist()


# In[63]:

with open('../data/total_results/new_labeled_sum.csv','r') as f:
    dddd = f.read()
with open('../data/total_results/new_labeled_sum.csv','w') as f:
    f.write('index,text\n')
    f.write(dddd)
r = pd.read_csv('/home/works/yexin/yexin/catagory/data/total_results/new_labeled_sum.csv')
c = []
final_ind = []
final_results = []
r_ind = list(r['index'])
for i in range(len(r_ind)):
    if int(r_ind[i]) in ind:
        continue
    final_ind.append(r_ind[i])
    final_results.append(r.loc[i,'text'])
df = pd.DataFrame({'index':final_ind, 'text': final_results})
df.to_csv('final_al_results.csv',index = False)

