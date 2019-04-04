#! /home/hmeng/anaconda3/bin/python

#KNN

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split


fruits_df = pd.read_table('data/fruit_data_with_colors.txt')
fruit_name_dict = dict(zip(fruits_df['fruit_label'], fruits_df['fruit_name']))
X = fruits_df[['mass', 'width', 'height', 'color_score']]
y = fruits_df['fruit_label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1/4, random_state=0)

from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)


from sklearn.metrics import accuracy_score

acc = accuracy_score(y_test, y_pred)
print('准确率：', acc)

k_range = range(1, 20)
acc_scores = []

for k in k_range:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    acc_scores.append(knn.score(X_test, y_test))

plt.figure()
plt.xlabel('k')
plt.ylabel('accuracy')
plt.scatter(k_range, acc_scores)
plt.xticks([0, 5, 10, 15, 20])
plt.show()
