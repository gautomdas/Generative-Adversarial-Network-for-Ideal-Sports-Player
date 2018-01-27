import pandas as pd
import os
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix


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

df =  pd.read_csv(os.path.join(data_dir, 'baseballClean.csv'))

print(df.columns)
df['label'].unique()

data = df[['playerid', 'WAR', 'height', 'weight', 'z_score', 'BMI']]
target = df['label']

data_train, data_test, target_train, target_test = train_test_split(data,target,test_size = 0.33,random_state=123)

simpleTree = DecisionTreeClassifier(max_depth=5)
simpleTree.fit(data_train,target_train)

gbmTree = GradientBoostingClassifier(max_depth=5)
gbmTree.fit(data_train,target_train)

rfTree = RandomForestClassifier(max_depth=5)
rfTree.fit(data_train,target_train)

simpleTreePerformance = precision_recall_fscore_support(target_test,simpleTree.predict(data_test))
gbmTreePerformance = precision_recall_fscore_support(target_test,gbmTree.predict(data_test))
rfTreePerformance = precision_recall_fscore_support(target_test,rfTree.predict(data_test))

print('Precision, Recall, Fscore, and Support for each class in simple, gradient boosted, and random forest tree classifiers:'+'\n')
for treeMethod in [simpleTreePerformance,gbmTreePerformance,rfTreePerformance]:
    print('Precision: ',treeMethod[0])
    print('Recall: ',treeMethod[1])
    print('Fscore: ',treeMethod[2])
    print('Support: ',treeMethod[3],'\n')

print('Confusion Matrix for simple, gradient boosted, and random forest tree classifiers:')
print('Simple Tree:\n',confusion_matrix(target_test,simpleTree.predict(data_test)),'\n')
print('Gradient Boosted:\n',confusion_matrix(target_test,gbmTree.predict(data_test)),'\n')
print('Random Forest:\n',confusion_matrix(target_test,rfTree.predict(data_test)))


print('Feature Importances for GBM tree\n')
for importance,feature in zip(gbmTree.feature_importances_,['playerid', 'WAR', 'height', 'weight', 'z_score', 'BMI']):
    print('{}: {}'.format(feature,importance))