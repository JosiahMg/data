#!/home/hmeng/anaconda3/bin/python

#list
print([1, 2] + [3, 4])
print([1]*5)
print(2 in [1, 2, 5])
l = [3, 5]
l.append(6)
print(l)

# tuple
t = (1, 'a', 2, 'b')
a, b, c, d = t
print(b)

#dictorary

dict = {'name':'Josh',
        'age': 15,
        'weight':'40kg'}

for key in dict.keys():
    print(key)

for value in dict.values():
    print(value)

for key, value in dict.items():
    print(key, value)

# set
my_set = {1, 2, 3}
print(type(my_set))
my_set = set([1, 2, 3, 2])
print(my_set)
my_set.add(4)
my_set.update([4, 5, 6])
print(my_set)


#string

s2 = 'Python is a widely used high-level programming language for general-purpose programming.'
print(s2.split(' '))


#time
import datetime
import time
   # from 1970/1/1
timestamp = time.time()

dt_now = datetime.datetime.fromtimestamp(timestamp)
print(dt_now)
print('{}年{}月{}日'.format(dt_now.year, dt_now.month, dt_now.day))

today = datetime.datetime.today().strftime('%y%m%d')
print(today)

#map
l1 = [1, 3, 5, 7, 9]
l2 = [2, 4, 6, 8, 10]
mins = map(min, l1, l2)
print(list(mins))

#lambda
l1 = [1, 3, 5, 7, 9]
l2 = [2, 4, 6, 8, 10]
result = map(lambda x, y: x * 2 + y, l1, l2)
print(list(result))

# list
l1 = [i for i in range(100) if i%2==0]



#csv
import csv

with open('./data/grades.csv') as csvfile:
    grades_data = list(csv.DictReader(csvfile))
    print(grades_data[0].keys())

sum_assign1 = sum(float(row['assignment1_grade']) for row in grades_data) / len(grades_data)
avg_assign1 = sum_assign1/len(grades_data)
print('assignment1平均分数：', avg_assign1)


# pack & unpack

l1 = list(range(1, 6))
l2 = list(range(6, 11))
zip_gen = zip(l1, l2)
x, y = zip(*zip_gen)
list(x)
