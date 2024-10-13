import os
import re
from bs4 import BeautifulSoup
import pandas as pd

# 指定 HTML 文件路径
path = "D:\\Tencent\\Downloads\\虚假新闻检测\\虚假新闻检测\\train\\html"


def getHtml(url):
    # 读取并解析 HTML 文件
    with open(url, 'r', encoding='utf-8') as file:
        soup = BeautifulSoup(file, 'html.parser')
    # 查找所有 <p> 标签
    data = ''
    paragraphs = soup.find_all('p')
    # 输出每个 <p> 标签的内容
    for p in paragraphs:
        data_temp = p.getText().replace('\n', '。')
        data += data_temp
        data += '。'
    return data


def getPathHtml():
    datalist = []
    for i in os.listdir(path):
        datalist.append(getHtml(path + "\\" + i))
    return datalist


def save2csv(path_csv, datalist):
    df = pd.read_csv(path_csv)
    df['News Url'] = datalist
    df.to_csv(path_csv, index=False)


if __name__ == "__main__":  # 当程序执行时
    datalist = getPathHtml()
    save2csv("D:\\Tencent\\Downloads\\虚假新闻检测\\虚假新闻检测\\train\\table.csv", datalist)
