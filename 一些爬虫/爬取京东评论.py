import requests
import json
import csv
import emoji

with open('commtent.csv', 'a', newline='', encoding='gbk') as fp:
    writer = csv.writer(fp)
    headers = ['id', 'nickname', 'content', 'time', 'score']
    writer.writerow(headers)
    first = 0
    #评论共有80页
    for i in range(0, 80):
        url = 'https://api.m.jd.com/?appid=item-v3&functionId=pc_club_productPageComments&client=pc&clientVersion=1.0.0&t=1700644803994&loginType=3&uuid=122270672.789545891.1699690539.1700636130.1700644724.6&productId=28105993532&score=0&sortType=5&page='
        finalurl = url + str(i) + '&pageSize=10&isShadowSku=0&fold=1&bbtf=&shield='
        header = {
            'User-Agent' :'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/118.0.0.0 Safari/537.36 Edg/118.0.2088.76'
                  }
        data = requests.get(url=finalurl,headers=header).text
        comment_list = json.loads(data)['comments']

        for j in range(len(comment_list)):
            id = comment_list[j]['id']
            nickname = comment_list[j]['nickname']
            content = comment_list[j]['content']
            content = emoji.demojize(content)
            time = comment_list[j]['creationTime'] #删除emo表情
            score = comment_list[j]['score']
            writer.writerow([id, nickname, content,time, score])

