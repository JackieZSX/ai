import os
import re
from bs4 import BeautifulSoup
import pandas as pd

# 指定 HTML 文件路径
path = ".\\train"


def getHtml(url):
    # 读取并解析 HTML 文件
    with open(url, 'r', encoding='utf-8') as file:
        soup = BeautifulSoup(file, 'html.parser')
    # 查找所有 <p> 标签
    data = ''
    paragraphs = soup.find_all('p')
    # 输出每个 <p> 标签的内容
    for p in paragraphs:
        data_temp = p.getText().replace('\n', ' ')
        if data_temp != '':
            data += data_temp
    return data


def getPathHtml():
    datalist = []
    htmlpath = path+'\\html'
    htmlpath_files = os.listdir(htmlpath)
    sorted_files = sorted(htmlpath_files, key=lambda x: int(x.split('.')[0]))
    for file in sorted_files:
        datalist.append(getHtml(htmlpath + "\\" + file))

    return datalist


def save2csv(path_csv, datalist):
    df = pd.read_csv(path_csv)
    df['News Url'] = datalist
    df.to_csv(path_csv, index=False)


if __name__ == "__main__":  # 当程序执行时

    # htmlpath = path + "\\html"
    # for i in os.listdir(htmlpath):
    #     print(i)
    # 获取目录下的所有文件和文件夹名
    # files = os.listdir(path+'\\html\\')
    # sorted_files = sorted(files, key=lambda x: int(x.split('.')[0]))
    # for file in sorted_files:
    #     print(file)
    # os.getdir(path)
    # data = getHtml(path+'\\html\\25.html')
    # print(data)
    datalist = getPathHtml()
    save2csv(path + "\\table.csv", datalist)
    # for i in os.listdir(path+'\\html\\'):
    #     print(i)