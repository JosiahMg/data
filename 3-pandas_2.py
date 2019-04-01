#! /home/hmeng/anaconda3/bin/python
import pandas as pd
import numpy as np


staff_df = pd.DataFrame([{'姓名': '张三', '部门': '研发部'},
                         {'姓名': '李四', '部门': '财务部'},
                         {'姓名': '赵六', '部门': '市场部'}])

student_df = pd.DataFrame([{'姓名': '张三', '专业': '计算机'},
                        {'姓名': '李四', '专业': '会计'},
                        {'姓名': '王五', '专业': '市场营销'}])

# how = outer / inner / left / right
pd.merge(staff_df, student_df, how='outer', on='姓名')


staff_df.rename(columns={'姓名':'员工姓名'}, inplace=True)
student_df.rename(columns={'姓名':'学生姓名'}, inplace=True)

staff_df['地址'] = ['天津', '北京', '上海']
student_df['地址'] = ['天津', '上海', '广州']

pd.merge(staff_df, student_df, how='left', left_on='员工姓名', right_on='学生姓名', suffixes=('(公司)', '(家乡)'))

staff_df['员工姓名'].apply(lambda x:x[0])



report_data = pd.read_csv('data/2015.csv')
groupby = report_data.groupby('Region')
groupby.size()

for group, frame in groupby:
    mean_score = frame['Happiness Score'].mean()
    max_score = frame['Happiness Score'].max()
    min_score = frame['Happiness Score'].min()
    print('{}地区的平均幸福指数：{}，最高幸福指数：{}，最低幸福指数{}'.format(group, mean_score, max_score, min_score))


cars_df = pd.read_csv('data/cars.csv')
cars_df.columns.tolist()
cars_df.pivot_table(values='(kW)', index='YEAR', columns='Make', aggfunc=np.mean)


np.random.binomial(100, 0.6)
np.random.normal(loc=1, size=100)
