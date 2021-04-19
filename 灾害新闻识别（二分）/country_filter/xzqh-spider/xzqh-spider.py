# -*- coding:utf-8 -*-
# editor: zzh

import requests
from bs4 import BeautifulSoup
import time
import hashlib

#无用代码，还是留着了

def get_response(link):
    headers = {
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_14_5) AppleWebKit/537.36 (KHTML, like Gecko) '
                          'Chrome/75.0.3770.100 Safari/537.36'
        }
    re = requests.get(link,headers=headers)
    if re.status_code == 200:
        realencoding = requests.utils.get_encodings_from_content(re.text)
        re.encoding = realencoding[0]
    return re.text

def hrefs_extract(text):
    """
    :param text: html文本
    :return: 所有国家对应href
    """

    soup = BeautifulSoup(text,features='lxml')
    xp = """p a"""
    infos = soup.select(xp)
    links = [info.get('href') for info in infos[:198]]
    return links

def get_all_wgxzqh(links):
    head = "http://www.xzqh.org/old/waiguo/"
    xp1 = """table > tr > td:nth-of-type(2)"""
    res = []
    for index,link in enumerate(links):
        try:
            text = get_response(head+link)
            soup = BeautifulSoup(text,features='lxml')
            infos = soup.select(xp1)
            cs = [info.text for info in infos]
            print(cs)
            res.append(cs)
        except Exception as e:
            e.with_traceback()
            pass
    return res
if __name__ == '__main__':
    text = get_response("""http://www.xzqh.org/old/waiguo/index.htm""")
    links = hrefs_extract(text)
    infos = get_all_wgxzqh(links)
    for info in infos:
        print(info)
