# -*- coding:utf-8 -*-
# editor: zzh
# date: 2021/1/5

import pickle
import re
from bs4 import BeautifulSoup

# with open("nses\\LocList.xml",'r',encoding='utf8') as f:
#     text = f.read()
#     soup = BeautifulSoup(text,features='lxml')
# # print(soup)
# countries = soup.select('countryregion')
# for country in countries[1:]:
#     print(country.get("name"))
#     states = country.select('state')
#     for state in states:
#         print(state.get("name"))
#         cities = state.select('city')
#         for city in cities:
#             print(city.get("name"))
#             regions = city.select('region')
#             for region in regions:
#                 print(region.get("name"))

# cities_in_china = pickle.load(open("cities_in_china.pk",'rb'))
# print(len(cities_in_china))
# with open("nses\\china_xzqh.txt",'r',encoding='utf8') as f:
#     lines = f.readlines()
#     for line in lines:
#         cs = re.split('、',line)
#         for c in cs:
#             c = c.strip()
#             c = c.replace(" ","")
#             c = re.sub("[省|市|县|区]", "", c)
#             print(c)
#             cities_in_china.add(c)
# cs = "西南，西北，东南，东北，西部，东部，中部，南部，北部"
# cs = cs.split('，')
# for c in cs:
#     cities_in_china.add(c)
# pickle.dump(cities_in_china,open("cities_in_china.pk",'wb'))
# print(len(cities_in_china))


# cities_in_waiguo = pickle.load(open("countries.pk",'rb'))
# print(len(cities_in_waiguo))
# with open("nses\\waiguo_xzqh.txt",'r',encoding='utf8') as f:
#     lines = f.readlines()
#     for line in lines:
#         cs = re.split('、',line)
#         for c in cs:
#             c = c.strip()
#             c = c.replace(" ","")
#             c = re.sub("[省|市|县|区]", "", c)
#             # print(c)
#             cities_in_waiguo.add(c)
# pickle.dump(cities_in_waiguo,open("countries.pk",'wb'))
# print(len(cities_in_waiguo))
