import numpy as np
import pandas as pd
import itertools
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

df = pd.read_csv('E:\\Explore\\Fake News detector\\news\\news.csv')
print(df.head())
labels = df.label
print(labels.head())

x_train, xtest, y_train, ytest = train_test_split(df['text'], labels, test_size = 0.2, random_state=2)
