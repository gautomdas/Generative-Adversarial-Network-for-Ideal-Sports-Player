import os
import numpy as np
import pandas as pd
from scipy import ndimage as nd
import tensorflow as tf

#Batch creation functions from MNIST tutorials
def dense_to_one_hot(labels_dense, num_classes=11):
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
    print(file_array)
    input_row = np.array(file_array)
    #input_row = np.reshape(input_row, (-1, 1))
    print(input_row.tolist())
    temp.append(input_row)

train_x = np.stack(temp)

# Split 80 20
split_size = int(train_x.shape[0] * 0.8)

train_x, val_x = train_x[:split_size], train_x[split_size:]
train_y, val_y = train.label.values[:split_size], train.label.values[split_size:]

#Test File
temp = []
for img_name in test.combined_data:
    file_array = row.split("_")
    count = 0
    for each_val in file_array:
        file_array[count] = float(each_val)
        count += 1
    print(file_array)
    input_row = np.array(file_array)
    #input_row = np.reshape(input_row, (-1, 1))
    print(input_row.tolist())
    temp.append(input_row)

test_x = np.stack(temp)

#Mutable variables
n_nodes_hl1 = 6
n_nodes_hl2 = 8
n_nodes_hl3 = 10

n_classes = 11
batch_size = 120
input_num_units = 4
output_num_units = 11
learning_rate = 0.001
hm_epochs = 150

# Place holders
x = tf.placeholder(tf.float32, [None, input_num_units])
y = tf.placeholder(tf.float32, [None, output_num_units])

#Three layer neural network
def neural_network_model(data, keep_prob):
    hidden_1_layer = {'weights': tf.Variable(tf.random_normal([input_num_units, n_nodes_hl1])),
                      'biases': tf.Variable(tf.random_normal([n_nodes_hl1]))}
    hidden_2_layer = {'weights': tf.Variable(tf.random_normal([n_nodes_hl1, n_nodes_hl2])),
                      'biases': tf.Variable(tf.random_normal([n_nodes_hl2]))}
    hidden_3_layer = {'weights': tf.Variable(tf.random_normal([n_nodes_hl2, n_nodes_hl3])),
                      'biases': tf.Variable(tf.random_normal([n_nodes_hl3]))}
    output_layer = {'weights': tf.Variable(tf.random_normal([n_nodes_hl3, n_classes])),
                    'biases': tf.Variable(tf.random_normal([n_classes])), }

    l1 = tf.add(tf.matmul(data, hidden_1_layer['weights']), hidden_1_layer['biases'])
    l1 = tf.nn.relu(l1)

    l2 = tf.add(tf.matmul(l1, hidden_2_layer['weights']), hidden_2_layer['biases'])
    l2 = tf.nn.relu(l2)

    l3 = tf.add(tf.matmul(l2, hidden_3_layer['weights']), hidden_3_layer['biases'])
    l3 = tf.nn.relu(l3)

    dropout = tf.nn.dropout(l3, keep_prob)

    output_layer = tf.matmul(dropout, output_layer['weights']) + output_layer['biases']
    return output_layer


def train_neural_network(x):
    prediction = neural_network_model(x, 1.0)
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=y))
    optimizer = tf.train.AdagradOptimizer(learning_rate=learning_rate).minimize(cost)

    # Tensorboard Create a summary to monitor cost tensor
    tf.summary.scalar("loss", cost)
    summary_op = tf.summary.merge_all
    # Merge all summaries
    merged_summary_op = tf.summary.merge_all()

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for epoch in range(hm_epochs):
            avg_cost = 0
            total_batch = int(train.shape[0] / batch_size)
            for i in range(total_batch):
                epoch_x, epoch_y = batch_creator(batch_size, train_x.shape[0], 'train')
                _, c , summary= sess.run([optimizer, cost,merged_summary_op], feed_dict={x: epoch_x, y: epoch_y})
                avg_cost += c / total_batch

            print("Epoch:", (epoch + 1), "cost =", "{:.5f}".format(avg_cost))

        pred_temp = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
        accuracy = tf.reduce_mean(tf.cast(pred_temp, "float"))
        print("\nValidation Accuracy:", accuracy.eval({x: val_x.reshape(-1, input_num_units), y: dense_to_one_hot(val_y)}))

        print("\nTraining complete!")

        predict = tf.argmax(prediction, 1)
        pred = predict.eval({x: test_x.reshape(-1, input_num_units)})
        sample_submission.combined_data = test.combined_data
        sample_submission.label = pred
        sample_submission.to_csv(os.path.join(sub_dir, 'sub_cnn.csv'), index=False)

train_neural_network(x)

sample_dir = os.path.join(sub_dir, 'sub_cnn.csv')
test_dir = os.path.join(data_dir, 'baseballTest.csv')

guess = np.genfromtxt(sample_dir, delimiter=',')
correct = np.genfromtxt(test_dir, delimiter=',')

counter = -1
correct_vals = 0
for each_row in guess:
    print()
    if(each_row[1] == int(correct[counter][1])):
        correct_vals += 1
    counter += 1

print(correct_vals/counter * 100)