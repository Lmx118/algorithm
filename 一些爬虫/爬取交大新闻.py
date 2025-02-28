import requests
from bs4 import BeautifulSoup

headers = {
  'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 '
                '(KHTML, like Gecko) Chrome/88.0.4324.190 Safari/537.36'
}


url='http://news.xjtu.edu.cn/zyxw.htm'
response=requests.get(url,headers=headers)

soup=BeautifulSoup(response.content,'html.parser')
news_list=soup.find_all('li',class_="clearfix")

news_content=[]
for news in news_list:
    news_url=news.find('a')['href']
    news_title=news.find('a')['title']
    url_new="https://news.xjtu.edu.cn/"+news_url
    news_response=requests.get(url_new,headers=headers)
    news_soup=BeautifulSoup(news_response.content,'html.parser')
    news_content_detail=news_soup.find_all('div',class_='v_news_content')[0].text.strip()
    news_content.append({'title':news_title,'content':news_content_detail})

for i in range(0,8):
    urls="http://news.xjtu.edu.cn/zyxw/"+str(1710-i)+".htm"
    response = requests.get(urls, headers=headers)

    soup = BeautifulSoup(response.content, 'html.parser')
    news_list = soup.find_all('li', class_="clearfix")

    for news in news_list:
        news_url = news.find('a')['href']
        news_title = news.find('a')['title']
        url_new = "https://news.xjtu.edu.cn/" + news_url
        news_response = requests.get(url_new, headers=headers)
        news_soup = BeautifulSoup(news_response.content, 'html.parser')
        news_content_detail = news_soup.find_all('div', class_='v_news_content')[0].text.strip()
        news_content.append({'title': news_title, 'content': news_content_detail})

with open('xjtunews.txt','w',encoding='utf-8') as f:
    for news in news_content:
        f.write(f"{news['title']}{news['content']}")
