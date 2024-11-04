import csv
import pandas as pd
import re

pattern = re.compile(
    r'^(?: 我要申诉 微信公众平台运营中心 | |接相关投诉，此内容违反《互联网用户公众账号信息服务管理规定》，查看详细内容 |nan|^$)')


def get_csv_inf():
    df = pd.read_csv('./train/table.csv')

    get_value = df['News Url']
    empty_array = [index for index, value in enumerate(get_value) if pattern.match(str(value))]

    get_id = df['id']

    get_office = df['Ofiicial Account Name']
    get_title = df['Title']

    get_labels = df['label']
    return empty_array, get_id, get_office, get_title, get_value, get_labels


def save_csv(info):
    empty_array, get_id, get_office, get_title, get_value, get_labels = info
    with open('./train/table_clean_out.csv', 'w', newline='', encoding="utf-8") as f:
        writer = csv.writer(f)
        csv_id = [value for index, value in enumerate(get_id) if index in empty_array]
        csv_office = [value for index, value in enumerate(get_office) if index in empty_array]
        csv_title = [value for index, value in enumerate(get_title) if index in empty_array]
        csv_news = [value for index, value in enumerate(get_value) if index in empty_array]
        csv_label = [value for index, value in enumerate(get_labels) if index in empty_array]

        writer.writerow(["id", "office", "title", "News", "label"])
        for i in range(len(csv_id)):
            writer.writerow([csv_id[i], csv_office[i], csv_title[i], csv_news[i], csv_label[i]])


if __name__ == '__main__':
    infomation = get_csv_inf()
    save_csv(infomation)
