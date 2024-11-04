import csv
import pandas as pd

df = pd.read_csv('./train/word_index.csv')
a = df['one_hot_vector']
print(a[0])

print(len(a[0]))
