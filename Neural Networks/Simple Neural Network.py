
"""
Created on Fri Nov  3 19:37:44 2017
@author: Gautom Das
"""
import os
import numpy as np
import pandas as pd
from scipy import ndimage as nd
import tensorflow as tf

#Batch creation functions from MNIST tutorials
def dense_to_one_hot(labels_dense, num_classes=10):
    num_labels = labels_dense.shape[0]
    index_offset = np.arange(num_labels) * num_classes
    labels_one_hot = np.zeros((num_labels, num_classes))
    labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1
    return labels_one_hot

def preproc(unclean_batch_x):
    temp_batch = unclean_batch_x / unclean_batch_x.max()
    return temp_batch

def batch_creator(batch_size, dataset_length, dataset_name):
    batch_mask = rng.choice(dataset_length, batch_size)
    batch_x = eval(dataset_name + '_x')[[batch_mask]].reshape(-1, input_num_units)
    batch_x = preproc(batch_x)
    if dataset_name == 'train':
        batch_y = eval(dataset_name).ix[batch_mask, 'label'].values
        batch_y = dense_to_one_hot(batch_y)
    return batch_x, batch_y

#Randomize learning
seed = 128
rng = np.random.RandomState(seed)

"""
All of the following rows of the program deal with file handling. 
"""
root_dir = os.path.abspath('..')
print(root_dir)
print(os.listdir(root_dir))

data_dir = os.path.join(root_dir, 'CSV Data Files')
curr_dir = os.path.join(root_dir, 'Neural Networks')
sub_dir = os.path.join(curr_dir, 'Submission Files')

# check for existence
print(os.path.exists(root_dir))
print(os.path.exists(data_dir))
print(os.path.exists(sub_dir))


train = pd.read_csv(os.path.join(data_dir, 'baseballClean.csv'))
test = pd.read_csv(os.path.join(data_dir, 'baseballTest.csv'))
sample_submission = pd.read_csv(os.path.join(sub_dir, 'Sample_Submission.csv'))

train.head()

temp = []
for row in train.combined_data:
    file_array = row.split("_")
    count = 0
    for each_val in file_array:
        file_array[count] = float(each_val)
        count+= 1
    print("_________________________")
    print(file_array)
    input_row = np.array(file_array)
    input_row = np.reshape(input_row, (-1, 1))
    temp.append(input_row)

train_x = np.stack(temp)

# Split 80 20
split_size = int(train_x.shape[0] * 0.8)

train_x, val_x = train_x[:split_size], train_x[split_size:]
train_y, val_y = train.label.values[:split_size], train.label.values[split_size:]

#Test File
temp = []
for row in test.combined_data:
    file_array = row.split("_")
    count = 0
    for each_val in file_array:
        file_array[count] = float(each_val)
        count += 1
    print(file_array)
    input_row = np.array(file_array)
    input_row = np.reshape(input_row, (-1, 1))
    temp.append(input_row)

test_x = np.stack(temp)

##Mutatable Parameters
# number of neurons in each layer
input_num_units = 4
hidden_num_units = 10 #width of the model
output_num_units = 10
# set remaining variables
epochs = 50
batch_size = 100#128
learning_rate = 0.001
class_num = output_num_units

# Place holders
x = tf.placeholder(tf.float32, [None, input_num_units])
y = tf.placeholder(tf.float32, [None, output_num_units])

#Weights and biases
weights = {
    'hidden': tf.Variable(tf.random_normal([input_num_units, hidden_num_units], seed=seed)),
    'output': tf.Variable(tf.random_normal([hidden_num_units, output_num_units], seed=seed))
}

biases = {
    'hidden': tf.Variable(tf.random_normal([hidden_num_units], seed=seed)),
    'output': tf.Variable(tf.random_normal([output_num_units], seed=seed))
}


#Single Layer Neural Network
hidden_layer = tf.add(tf.matmul(x, weights['hidden']), biases['hidden'])
hidden_layer = tf.nn.relu(hidden_layer)

output_layer = tf.matmul(hidden_layer, weights['output']) + biases['output']

# Back propagation
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=output_layer, labels=y))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(cost)

#Init variables
init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    for epoch in range(epochs):
        avg_cost = 0
        total_batch = int(train.shape[0] / batch_size)
        for i in range(total_batch):
            batch_x, batch_y = batch_creator(batch_size, train_x.shape[0], 'train')
            _, c = sess.run([optimizer, cost], feed_dict={x: batch_x, y: batch_y})

            avg_cost += c / total_batch

        print("Epoch:", (epoch + 1), "cost =", "{:.5f}".format(avg_cost))


    print("\nTraining complete!")

    pred_temp = tf.equal(tf.argmax(output_layer, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(pred_temp, "float"))
    print("Validation Accuracy:", accuracy.eval({x: val_x.reshape(-1, input_num_units), y: dense_to_one_hot(val_y)}))

    predict = tf.argmax(output_layer, 1)
    pred = predict.eval({x: test_x.reshape(-1, input_num_units)})
    sample_submission.combined_data = test.combined_data
    sample_submission.label = pred
    sample_submission.to_csv(os.path.join(sub_dir, 'sub_rmlp.csv'), index=False)

sample_dir = os.path.join(sub_dir, 'sub_rmlp.csv')
test_dir = os.path.join(data_dir, 'baseballTest.csv')

guess = np.genfromtxt(sample_dir, delimiter=',')
correct = np.genfromtxt(test_dir, delimiter=',')

counter = 0
correct_vals = 0
for each_row in guess:
    if(each_row[1] == correct[counter][1]):
        correct_vals += 1
    counter += 1

print(correct_vals/counter * 100)
