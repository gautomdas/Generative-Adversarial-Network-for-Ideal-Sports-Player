import os
import csv

root_dir = os.path.abspath('./')
other_dir = os.path.abspath('..')
print(root_dir)
print(other_dir)


data_dir = os.path.join(other_dir, 'CSV Data Files')
curr_dir = os.path.join(root_dir, 'Neural Networks')
sub_dir = os.path.join(curr_dir, 'Submission Files')

# check for existence
print(os.path.exists(root_dir))
print(os.path.exists(curr_dir))
print(os.path.exists(sub_dir))


train = (os.path.join(data_dir, 'baseballClean.csv'))
test = (os.path.join(sub_dir, 'baseballTest.csv'))

baseball_data = []
with open(train, newline='') as myFile:
    reader = csv.reader(myFile)
    for row in reader:
        baseball_data.append(row)

    print(baseball_data)

submit_array = []
counter = 3
reg_array = []
for eachPass in range(0, 6):
    reg_array.append(baseball_data[0][counter])
    counter += 1
submit_array.append(reg_array)

baseball_data.pop(0)
print(baseball_data)
print(submit_array)


counte = 0
reg_array = []
for eachPass in range(0, len(baseball_data)):
    counter = 3
    for eachPass in range(0, 6):
        reg_array.append(baseball_data[counte][counter])
        counter += 1
    counte+= 1
    submit_array.append(reg_array)
    print(len(submit_array))

fin_array = []
for eachPass in range(0, len(baseball_data)):
    counter = 3
    for eachPass in range(0, 6):
        reg_array.append(baseball_data[counte][counter])
        counter += 1
    counte+= 1
    submit_array.append(reg_array)
    print(len(submit_array))