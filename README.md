# ai

# my_train.csv

该文件为清洗的训练的特征向量：

清洗内容包括：
· html文字提取
· 筛选文章已被删除的数据
· 替换文章无用词

特征向量：
· id：train.csv中的id，原数据唯一标识
· text_length：文本长度
· title_length：标题长度
· quote_symbol_rate：文章的所有符号中引号占比（已归一化）
· strong_emotion_symbol_rate：文章的所有符号中感叹号和问号占比（已归一化）
· source_from_entertainment：Ofiicial Account Name是否包含“娱乐”、“八卦”或“搞笑”
· title_contains_person：Title包含人名数量（已归一化）
                    
