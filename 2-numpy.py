#! /home/hmeng/anaconda3/bin/python

import numpy as np

my_list = [1, 2, 3, 4]
x = np.array(my_list)

n = np.array([[1, 2, 3], [4, 5, 6]])


n = np.arange(0, 30, 2)
n = n.reshape(3, 5)

np.ones(3)
np.zeros(3)
np.eye(3)
np.diag(my_list)

np.array([1, 2, 3]*3)
np.repeat([1, 2, 3], 3)

p1 = np.ones((3, 3))
p2 = np.arange(9).reshape(3, 3)
np.vstack((p1, p2))
np.hstack((p1, p2))

p1+p2
p1*p2
p1**2
p1.dot(p2)


p3 = np.arange(6).reshape(2, 3)
p4 = p3.T
print(p4.dtype)
p5 = p3.astype('float')

a = np.array([-4, -2, 1, 3, 5])
print('sum: ', a.sum())
print('min: ', a.min())
print('max: ', a.max())
print('mean: ', a.mean())
print('std: ', a.std())
print('argmax: ', a.argmax())
print('argmin: ', a.argmin())

t = np.random.randint(0, 10, (4, 3))
for i, row in enumerate(t):
    print('row {} is {}'.format(i, row))

t2 = t**2
for i, j in zip(t, t2):
    print('{} + {} = {}'.format(i, j, i + j))
