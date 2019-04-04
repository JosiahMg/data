#! /home/hmeng/anaconda3/bin/python

import matplotlib.pyplot as plt
import numpy as np


plt.plot(3, 2, '*')

plt.figure()
plt.plot(4, 5, 'o')
ax = plt.gca()
ax.axis([0, 10, 0, 10])

x = np.array([1, 2, 3, 4, 5, 6, 7, 8])
y = x
colors = ['red'] * (len(x) - 1)
colors.append('green')
plt.scatter(x, y, s=100, c=colors)

plt.xlabel('x label')
plt.ylabel('y label')
plt.title('Scatter Plot Example')

plt.legend(loc=4, frameon=False, title='Legend')


linear_data = np.arange(1, 9)
quadratic_data = linear_data ** 2
plt.plot(linear_data, '-o', quadratic_data, '-o')

plt.gca().fill_between(range(len(linear_data)),
                      linear_data, quadratic_data,
                      facecolor='green',
                      alpha=0.25)

observation_dates = np.arange('2017-10-11', '2017-10-19', dtype='datetime64[D]')
observation_dates = list(map(pd.to_datetime, observation_dates))
plt.plot(observation_dates, linear_data, '-o',
        observation_dates, quadratic_data, '-o')

ax = plt.gca()
ax.set_title('Quadratic ($x^2$) vs. Linear ($x$)')


x_vals = list(range(len(linear_data)))
plt.bar(x_vals, linear_data, width=0.3)

x_vals2 = [item + 0.3 for item in x_vals]
plt.bar(x_vals2, quadratic_data, width=0.3)


x_vals = list(range(len(linear_data)))
plt.barh(x_vals, linear_data, height=0.3)
plt.barh(x_vals, quadratic_data, height=0.3, left=linear_data)


ax1 = plt.subplot(1, 2, 1)
plt.plot(linear_data, '-o')
ax2 = plt.subplot(1, 2, 2, sharey=ax1)
plt.plot(exponential_data, '-x')


fig, ((ax1, ax2, ax3), (ax4, ax5, ax6)) = plt.subplots(2, 3, sharex=True, sharey=True)
ax5.plot(exponential_data, '-')

fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, sharex=True)
axs = [ax1, ax2, ax3, ax4]

for n in range(len(axs)):
    sample_size = 10 ** (n + 1)
    sample = np.random.normal(loc=0., scale=1., size=sample_size)
    # 默认bin的个数为10
    axs[n].hist(sample)
    axs[n].set_title('n={}'.format(sample_size))

y = np.random.normal(loc=0., scale=1., size=10000)
x = np.random.random(size=10000)
plt.hist2d(x, y, bins=100)
plt.colorbar()


# 正态分布采样
normal_sample = np.random.normal(loc=0., scale=1., size=10000)
# 随机数采样
random_sample = np.random.random(size=10000)
# gamma分布采样
gamma_sample = np.random.gamma(2, size=10000)

df = pd.DataFrame({'normal': normal_sample,
                  'random': random_sample,
                  'gamma': gamma_sample})

plt.figure()
plt.boxplot([df['normal'], df['random'], df['gamma']], whis='range')
