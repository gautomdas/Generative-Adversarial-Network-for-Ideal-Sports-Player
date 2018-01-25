"""
Created on Fri Nov  3 19:37:44 2017
@author: Gautom Das
"""

#Initial imports
from __future__ import print_function
import tensorflow as tf
from tensorflow.contrib import rnn
import os
import pandas as pd
import numpy as np
import scipy.ndimage as nd

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
root_dir = os.path.abspath('.')
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

# Split 70:30
split_size = int(train_x.shape[0] * 0.7)

train_x, val_x = train_x[:split_size], train_x[split_size:]
train_y, val_y = train.label.values[:split_size], train.label.values[split_size:]

##Mutable Parameters
# Training Parameters
learning_rate = 0.01
training_steps = 50 #epoch
batch_size = 10
display_step = 5
# Network Parameters
num_input = 4
timesteps = 4
num_hidden = 10
num_classes = 10
input_num_units = 5

# tf Graph input
x = tf.placeholder("float", [None, timesteps, num_input])
y = tf.placeholder("float", [None, num_classes])

# Weights and biases
weights = {
    'out': tf.Variable(tf.random_normal([num_hidden, num_classes]))
}
biases = {
    'out': tf.Variable(tf.random_normal([num_classes]))
}

keep_prob = tf.placeholder(tf.float32)


def RNN(x, weights, biases):
    x = tf.unstack(x, timesteps, 1)

    lstm_cell = rnn.BasicLSTMCell(num_hidden, forget_bias=1)
    outputs, states = rnn.static_rnn(lstm_cell, x, dtype=tf.float32)

    return tf.matmul(outputs[-1], weights['out']) + biases['out']

#To run model
logits = RNN(x, weights, biases)
prediction = tf.nn.softmax(logits)

loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y))
optimizer = tf.train.AdagradOptimizer(learning_rate=learning_rate)
train_op = optimizer.minimize(loss_op)


correct_pred = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# Tensorboard Create a summary to monitor cost tensor
tf.summary.scalar("loss", loss_op)
summary_op = tf.summary.merge_all
# Merge all summaries
merged_summary_op = tf.summary.merge_all()

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    counter = 0
    avg_acc = 0.0
    sum = 0.0
    for step in range(1, training_steps + 1):
        total_batch = int(train.shape[0] / batch_size)
        batch_x, batch_y = batch_creator(batch_size, train_x.shape[0], 'train')
        batch_x = batch_x.reshape((batch_size, timesteps, num_input))

        sess.run(train_op, feed_dict={x: batch_x, y: batch_y})
        if step % display_step == 0 or step == 1:
            counter += 1
            loss, acc , summary= sess.run([loss_op, accuracy, merged_summary_op], feed_dict={x: batch_x, y: batch_y})

            print("Step " + str(step) + ", Training Accuracy= " + "{:.6f}".format(acc))
            sum += float(acc)


    summary_writer = tf.summary.FileWriter("output", sess.graph)

    print("Optimization Finished!")

    pred_temp = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(pred_temp, "float"))

    print("\nTraining complete!")

    predict = tf.argmax(prediction, 1)
    pred = predict.eval({x: test_x.reshape(-1, 1408, num_input)})
    sample_submission.filename = test.filename
    sample_submission.label = pred
    sample_submission.to_csv(os.path.join(sub_dir, 'sub_lstm.csv'), index=False)

    print("\nSample File complete!")

sample_dir = os.path.join(sub_dir, 'sub_lstm.csv')
test_dir = os.path.join(sub_dir, 'test_image.csv')

guess = np.genfromtxt(sample_dir, delimiter=',')
correct = np.genfromtxt(test_dir, delimiter=',')

counter = 0
correct_vals = 0
for each_row in guess:
    if (each_row[1] == correct[counter][1]):
        correct_vals += 1
    counter += 1

acc = (correct_vals / counter * 100)
print("\nCalculated Accuracy : ", acc)

