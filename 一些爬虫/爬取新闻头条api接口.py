import urllib.parse, urllib.request, json
import csv
#定义了一个变量 url，表示要访问的 API 接口的 URL 地址
url = 'http://v.juhe.cn/toutiao/index'

#定义了一个字典类型的变量 params，包含了请求参数 type 和 key，type 表示要获取的新闻类型，key 是需要申请的接口API接口请求Key。
params = {
    "type": "top",
    "key": "88f52dea96e3e250614d94fad83c7ef5",
    "page": "1", #当前页数
    "page_size": "30", #每页返回条数
    "is_filter": "0",
}

querys = urllib.parse.urlencode(params).encode('utf-8')

request = urllib.request.Request(url, data=querys)

response = urllib.request.urlopen(request)
content = response.read()
with open('news.csv', 'a', newline='', encoding='gbk') as fp:
    writer = csv.writer(fp)
    headers = ['title', 'date', 'url']
    writer.writerow(headers)
    if (content):
        try:
            result = json.loads(content.decode('utf-8'))
            error_code = result['error_code']
            if (error_code == 0):
                data = result['result']['data']
                for i in data:
                    title=i['title']
                    date=i['date']
                    url=i['url']
                    writer.writerow([title, date, url])

            else:
                print("请求失败:%s %s" % (result['error_code'], result['reason']))
        except Exception as e:
            print("解析结果异常：%s" % e)
    else:
        # 可能网络异常等问题，无法获取返回内容，请求异常
        print("请求异常")

