import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn import tree
import joblib

data = pd.read_csv('D:/machine_learning/Machine_learning_training_n_testing_model/2_class_data.csv')

X = np.array(data[[
    'x1',
    'x2'
]])

y = np.array(data['y'])

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.5)

#Decision Tree Classifier
#classifier = DecisionTreeClassifier()
#classifier.fit(X_train,y_train)

#Saving model
#joblib.dump(classifier, '2_class_data.pkl')

#Loading model 
classifier_from_joblib = joblib.load('2_class_data.pkl')

#Using Saved model to make prediction
y_pred = classifier_from_joblib.predict(X_test)
print("Predicted Output:")
print(y_pred)

#Visualizing the decision tree model
#tree.plot_tree(classifier)


#Making Prediction
#y_pred = classifier.predict(X_test)
#print("Predicted Output:")
#print(y_pred)
acc = accuracy_score(y_test,y_pred)
print("Testing data:")
print(y_test)
print(acc)