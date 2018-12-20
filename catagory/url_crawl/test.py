# -*- coding: utf-8 -*-
import pickle
from url import Searcher
import time

results = Searcher.question_search_get('我想查一下')
questions = pickle.load(open('total_texts.txt', 'r'))
dic = {}
for i in range(len(questions)):
    if type(questions[i][0]) == float:
        s = ''
    else:
        s = questions[i][0]
    print(s)
    print(i)
    time.sleep(0.05)
    start = time.time()
    results = Searcher.question_search_get(s)
    end = time.time()
    print('time:' + str(end - start) + 's')
    dic[questions[i][1]] = results
pickle.dump(dic, open('dic_results_500.txt', 'w'))
