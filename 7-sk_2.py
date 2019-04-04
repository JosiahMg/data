#! /home/hmeng/anaconda3/bin/python

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier


import matplotlib.pyplot as plt
from sklearn.datasets import make_regression


plt.figure()
plt.title('Sample regression problem with one input variable')

# 每个样本只有一个变量  make data
X_R1, y_R1 = make_regression(n_samples = 100, n_features=1,
                            n_informative=1, bias = 150.0,
                            noise = 30, random_state=0)
plt.scatter(X_R1, y_R1, marker= 'o', s=50)

# LinearRegression
from sklearn.linear_model import LinearRegression

X_train, X_test, y_train, y_test = train_test_split(X_R1, y_R1,
                                                   random_state = 0)
# 调用线型回归模型
linreg = LinearRegression()

# 训练模型
linreg.fit(X_train, y_train)

# 输出结果
print('线型模型的系数(w): {}'.format(linreg.coef_))
print('线型模型的常数项(b): {:.3f}'.format(linreg.intercept_))
print('训练集中R-squared得分: {:.3f}'.format(linreg.score(X_train, y_train)))
print('测试集中R-squared得分: {:.3f}'.format(linreg.score(X_test, y_test)))



# 可视化输出结果

plt.figure(figsize=(5,4))
plt.scatter(X_R1, y_R1, marker= 'o', s=50, alpha=0.8)
plt.plot(X_R1, linreg.coef_ * X_R1 + linreg.intercept_, 'r-')
plt.title('Least-squares linear regression')
plt.xlabel('Feature value (x)')
plt.ylabel('Target value (y)')
#plt.show()


# LogisticRegression
from sklearn.linear_model import LogisticRegression
# 加载数据集
fruits_df = pd.read_table('data/fruit_data_with_colors.txt')

X = fruits_df[['width', 'height']]
y = fruits_df['fruit_label'].copy()

# 将不是apple的标签设为0
y[y != 1] = 0
# 分割数据集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1/4, random_state=0)

# 不同的C值
c_values = [0.1, 1, 100]

for c_value in c_values:
    # 建立模型
    lr_model = LogisticRegression(C=c_value)

    # 训练模型
    lr_model.fit(X_train, y_train)

    # 验证模型
    y_pred = lr_model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    print('C={}，准确率：{:.3f}'.format(c_value, acc))


# SVM
from sklearn.svm import SVC

# 加载数据集
fruits_df = pd.read_table('data/fruit_data_with_colors.txt')

X = fruits_df[['width', 'height']]
y = fruits_df['fruit_label'].copy()

# 将不是apple的标签设为0
y[y != 1] = 0
# 分割数据集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1/4, random_state=0)

# 不同的C值
c_values = [0.0001, 1, 10000]

for c_value in c_values:
    # 建立模型
    svm_model = SVC(C=c_value)

    # 训练模型
    svm_model.fit(X_train, y_train)

    # 验证模型
    y_pred = svm_model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    print('C={}，准确率：{:.3f}'.format(c_value, acc))





# DecisionTree

from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split

iris = load_iris()

X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, random_state=0)

max_depth_values = [2, 3, 4]

for max_depth_val in max_depth_values:
    dt_model = DecisionTreeClassifier(max_depth=max_depth_val)
    dt_model.fit(X_train, y_train)

    print('max_depth=', max_depth_val)
    print('训练集上的准确率: {:.3f}'.format(dt_model.score(X_train, y_train)))
    print('测试集的准确率: {:.3f}'.format(dt_model.score(X_test, y_test)))
    print()
