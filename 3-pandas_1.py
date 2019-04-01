#! /home/hmeng/anaconda3/bin/python

import pandas as pd

countries = ['China', 'America', 'Australia']
countires_s = pd.Series(countries)

country_dicts = {'CH':'China', 'US':'America', 'AU':'Australia'}
country_dicts_s = pd.Series(country_dicts)
print(country_dicts_s.index)
print(country_dicts_s['CH'])
print(country_dicts_s.iloc[1])
print(country_dicts_s.loc[['AU', 'CH']])


country1 = pd.Series({'Name': '中国',
                    'Language': 'Chinese',
                    'Area': '9.597M km2',
                     'Happiness Rank': 79})

country2 = pd.Series({'Name': '美国',
                    'Language': 'English (US)',
                    'Area': '9.834M km2',
                     'Happiness Rank': 14})

country3 = pd.Series({'Name': '澳大利亚',
                    'Language': 'English (AU)',
                    'Area': '7.692M km2',
                     'Happiness Rank': 9})

df = pd.DataFrame([country1, country2, country3], index=['CH', 'US', 'AU'])

df['Location'] = '地球'
df['Location']
df.loc['CH']


df.drop('CH')
df.drop('CH', inplace=True)

df.drop('Area', axis=1)

ranks = df['Happiness Rank']
ranks += 2

ranks = df['Happiness Rank'].copy()
ranks += 2

report_2015_df = pd.read_csv('data/2015.csv')
report_2015_df.head()

report_2016_df = pd.read_csv('data/2016.csv',
                             index_col='Country',
                             usecols=['Country', 'Happiness Rank', 'Happiness Score', 'Region'])

report_2016_df.columns
report_2016_df.index
report_2016_df.reset_index()

report_2016_df.rename(columns={'Region': '地区', 'Happiness Rank': '排名', 'Happiness Score': '幸福指数'}, inplace=True)


only_western_europe_10 = (report_2016_df['地区'] == 'Western Europe') & (report_2016_df['排名'] > 10)
report_2016_df[only_western_europe_10]

log_data = pd.read_csv('data/log.csv')
log_data.set_index(['time', 'user'], inplace=True)
log_data.sort_index(inplace=True)
log_data.fillna(0)
log_data.dropna()
log_data.ffill()
log_data.bfill()
