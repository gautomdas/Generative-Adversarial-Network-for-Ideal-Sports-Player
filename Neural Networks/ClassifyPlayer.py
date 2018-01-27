import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt


plt.style.use('ggplot') # make plots look better


#### import the data ####
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

df =  pd.read_csv(os.path.join(data_dir, 'baseballClean.csv'))
test = pd.read_csv(os.path.join(data_dir, 'baseballTest.csv'))
sample_submission = pd.read_csv(os.path.join(sub_dir, 'Sample_Submission.csv'))


print (df.head())
print (df.describe())


#### prepare data for sklearn ####
# drop irreleveant coloums
#df_feature_selected =df[['playerid', 'WAR', 'height', 'weight', 'z_score', 'BMI']]
#test_feature_selected =test[['playerid', 'WAR', 'height', 'weight', 'z_score', 'BMI']]

df_feature_selected =df[['playerid',  'height', 'weight',  'BMI']]
test_feature_selected =test[['playerid', 'height', 'weight', 'BMI']]

# create and encode labels
labels = np.asarray(df.label)

from sklearn import preprocessing
le = preprocessing.LabelEncoder()
le.fit(labels)

labels = le.transform(labels)

# create features using DictVectorizer, and pandas's to_dict method
df_features = df_feature_selected.to_dict( orient = 'records' )

from sklearn.feature_extraction import DictVectorizer
vec = DictVectorizer()
features = vec.fit_transform(df_features).toarray()


##### split up in test and training data ####
from sklearn.model_selection import train_test_split

features_train, features_test, labels_train, labels_test = train_test_split(
	features, labels,
	test_size=0.33, random_state=42)


#### Fit to random forests ####

# Random Forests Classifier
"""from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier(
	min_samples_split=4,
	criterion="entropy"
	)
"""
# Support Vector Classifier
"""
from sklearn.svm import SVC
clf = SVC()
"""

"""
from sklearn.tree import DecisionTreeClassifier
clf = DecisionTreeClassifier(max_depth=5)
"""

from sklearn.ensemble import GradientBoostingClassifier
clf = GradientBoostingClassifier(max_depth=3)

clf.fit(features_train, labels_train)

# find the accuracy of the model
acc_test = clf.score(features_test, labels_test)
acc_train = clf.score(features_train, labels_train)
print('---------------------------')

print ("Train Accuracy:", acc_train)
print ("Test Accuracy:", acc_test)


# compute predictions on test features
pred = clf.predict(features_test)

# predict our new player
#Player = [13611,5.3,69,180,2.385429648,26.57844991]

label_pred = clf.predict(test_feature_selected) # [1]
#
#print (label_pred) # [7]

#result = pd.concat([test_feature_selected, pred], axis=1)
sample_submission.combined_data =test_feature_selected.playerid
sample_submission.label = label_pred
sample_submission.to_csv(os.path.join(sub_dir, 'subplayer.csv'), index=False)

#### Figure out what kind of mistakes it makes ####
from sklearn.metrics import recall_score, precision_score

precision = precision_score(labels_test, pred, average="weighted")
recall = recall_score(labels_test, pred, average="weighted")

print('---------------------------')

print ("Precision:", precision) #false negatives
print ("Recall:", recall) #false positives

sample_dir = os.path.join(sub_dir, 'subplayer.csv')
test_dir = os.path.join(data_dir, 'baseballTest.csv')

guess = np.genfromtxt(sample_dir, delimiter=',', names=True)
correct = np.genfromtxt(test_dir, delimiter=',', names=True)
print('---------------------------')


counter = 0
correct_vals = 0
for each_row in guess:
	if(each_row['label'] == correct[counter]['label']):
		correct_vals = correct_vals + 1
	counter = counter + 1

acc = (correct_vals/counter * 100)
print("Validation Accuracy  - " , acc)