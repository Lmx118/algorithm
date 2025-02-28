import requests
from bs4 import BeautifulSoup
import csv

def request_data(url):
    headers = {
        'User-Agent' :'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/118.0.0.0 Safari/537.36 Edg/118.0.2088.76'
    }
    res = requests.get(url, headers=headers)
    html = res.content.decode('utf-8')
    return html

def parse_data(html):
    soup = BeautifulSoup(html, 'html5lib')
    div = soup.find(class_="conMidtab")
    li = []
    tables =div.find_all('table')
    for table in tables:
        tr_tag = table.find_all('tr')[2:]

        for index, tr in enumerate(tr_tag):
            a = {}
            tds = tr.find_all('td')
            if index == 0:
                cities = list(tds[1].stripped_strings)[0]
            else:
                cities = list(tds[0].stripped_strings)[0]
            temp1 = list(tds[-5].stripped_strings)[0] #倒数第5个td是最高气温
            temp2 = list(tds[-2].stripped_strings)[0] #倒数第2个td是最低气温

            a['city'] = cities
            a['high temp'] = temp1
            a['low temp'] = temp2
            li.append(a)

    return li

# 保存数据
def save_csv(li, header):
    with open('weather.csv', 'w', encoding='utf-8-sig', newline="") as file:
        writer = csv.DictWriter(file, header)
        writer.writerows(li)


if __name__ == '__main__':
    url = 'http://www.weather.com.cn/textFC/hb.shtml'
    b = request_data(url)
    li = parse_data(b)
    header = ('city', 'high temp','low temp')
    save_csv(li, header)
