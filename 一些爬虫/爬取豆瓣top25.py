import requests
from lxml import etree
import pandas as pd
import csv

headers={
    'User-Agent' :'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/118.0.0.0 Safari/537.36 Edg/118.0.2088.76'
}

url='https://movie.douban.com/top250'
response=requests.get(url,headers=headers)

text=response.text
html=etree.HTML(text)

lilist=html.xpath('//ol[@class="grid_view"]/li')
movies=[]
for li in lilist:
    title=li.xpath('.//span[@class="title"][1]/text()')[0]
    bd=li.xpath('.//div[@class="bd"]/p[1]/text()')
    directors_actors=bd[0].strip()
    split_info1 =directors_actors.split('\xa0\xa0\xa0')
    directors = split_info1[0].strip('导演: ')
    actors = split_info1[1].strip('主演: ')
    info2=bd[1].strip()
    split_info2 = info2.split('\xa0/\xa0')
    years = split_info2[0]
    score=li.xpath('.//span[@class="rating_num"]/text()')[0]
    num=li.xpath('.//div[@class="star"]/span[4]/text()')[0]
    urls=li.xpath('.//div[@class="hd"]/a/@href')
    movie={
        'title':title,
        'directors': directors,
        'actors':actors,
        'years':years,
        'score':score,
        'num':num,
        'urls':urls,
    }
    movies.append(movie)

header=['title','directors','actors','years','score','num','urls']
with open('movies.csv', 'w', encoding='utf-8-sig', newline="") as file:
    writer = csv.DictWriter(file, header)
    writer.writerows(movies)
