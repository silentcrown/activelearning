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
        sim_results = []
        if not rsp['errno'] == 0:
            print rsp
            return 
        sim_results = [x['info'][fields.split('|')[0]]
            for x in rsp['results']['results']]
        return sim_results