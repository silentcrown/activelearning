import json
import requests
from urllib import quote


class Searcher(object):
    _host = '172.16.11.111:2010'
    _timeout_s = 3

    @classmethod
    def question_search_get(cls, query, size=500):
        url = ('http://%s/saas/v1/201/search/knowledge?query=%s&size=%s'
            % (cls._host, query, size))
        r = requests.get(url)
        # print r.content
        rsp = r.json()
        results = []
        if not rsp['errno'] == 0:
            print rsp
            return 
        results = [(x['k_question'], x['score'])
            for x in rsp['data']['sim_questions']]
        return results

