#! /home/hmeng/anaconda3/bin/python

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import seaborn as sns
from sklearn.model_selection import train_test_split



fruits_df = pd.read_table('data/fruit_data_with_colors.txt')

fruit_name_dict = dict(zip(fruits_df['fruit_label'], fruits_df['fruit_name']))

X = fruits_df[['mass', 'width', 'height', 'color_score']]
y = fruits_df['fruit_label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1/4, random_state=0)
