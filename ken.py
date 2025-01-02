# #imrting necessary Libraries
# from sklearn import datasets
# from sklearn.metrics import confusion_matrix
# from sklearn.model_selection import train_test_split

# # Loading the iris dataset 
# iris = datasets.load_iris()

# # X-> features, y-> label
# X = iris.data
# y = iris.target 


# # Dividing X, y into train and test data 
# X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 0)

# # Training a DecisionTreeClassifier 
# from sklearn.tree import DecisionTreeClassifier
# dtree_model = DecisionTreeClassifier(max_depth = 2).fit(X_train, y_train)
# dtree_prediction = dtree_model.predict(X_test)

# # creating confusion matrix

# cm = confusion_matrix(y_test,dtree_prediction)

# # # importing libraries 
# import pandas as pd 
# import numpy as np 

# # read the data in a pandas dataframe 
# inp = np.array([[1., 2., 3., 4., 5.], [6., 7., 8., 9., 10.], [11., 12., 13., 14., 15.]])

# # drop or delete the unnecessary columns in the data. 
# data = data.drop(['Events', 'Date', 'SeaLevelPressureHighInches', 
# 				'SeaLevelPressureLowInches'], axis=1) 

# # some values have 'T' which denotes trace rainfall 
# # we need to replace all occurrences of T with 0 
# # so that we can use the data in our model 
# data = data.replace('T', 0.0) 

# # the data also contains '-' which indicates no 
# # or NIL. This means that data is not available 
# # we need to replace these values as well. 
# data = data.replace('-', 0.0) 

# # save the data in a csv file 
# data.to_csv('austin_final.csv') 


# # importing libraries 
# import pandas as pd 
# import numpy as np 
# import sklearn as sk 
# from sklearn.linear_model import LinearRegression 
# import matplotlib.pyplot as plt 

# # read the cleaned data 
# data = pd.read_csv("austin_final.csv") 

# # the features or the 'x' values of the data 
# # these columns are used to train the model 
# # the last column, i.e, precipitation column 
# # will serve as the label 
# X = data.drop(['PrecipitationSumInches'], axis=1) 

# # the output or the label. 
# Y = data['PrecipitationSumInches'] 
# # reshaping it into a 2-D vector 
# Y = Y.values.reshape(-1, 1) 

# # consider a random day in the dataset 
# # we shall plot a graph and observe this 
# # day 
# day_index = 798
# days = [i for i in range(Y.size)] 

# # initialize a linear regression classifier 
# clf = LinearRegression() 
# # train the classifier with our 
# # input data. 
# clf.fit(X, Y) 

# # give a sample input to test our model 
# # this is a 2-D vector that contains values 
# # for each column in the dataset. 
# inp = np.array([[74], [60], [45], [67], [49], [43], [33], [45], 
# 				[57], [29.68], [10], [7], [2], [0], [20], [4], [31]]) 
# inp = inp.reshape(1, -1) 

# # print the output. 
# print('The precipitation in inches for the input is:', clf.predict(inp)) 

# # plot a graph of the precipitation levels 
# # versus the total number of days. 
# # one day, which is in red, is 
# # tracked here. It has a precipitation 
# # of approx. 2 inches. 
# print("the precipitation trend graph: ") 
# plt.scatter(days, Y, color='g') 
# plt.scatter(days[day_index], Y[day_index], color='r') 
# plt.title("Precipitation level") 
# plt.xlabel("Days") 
# plt.ylabel("Precipitation in inches") 


# plt.show() 
# x_vis = X.filter(['TempAvgF', 'DewPointAvgF', 'HumidityAvgPercent', 
# 				'SeaLevelPressureAvgInches', 'VisibilityAvgMiles', 
# 				'WindAvgMPH'], axis=1) 

# # plot a graph with a few features (x values) 
# # against the precipitation or rainfall to observe 
# # the trends 

# print("Precipitation vs selected attributes graph: ") 

# for i in range(x_vis.columns.size): 
# 	plt.subplot(3, 2, i + 1) 
# 	plt.scatter(days, x_vis[x_vis.columns.values[i][:100]], 
# 				color='g') 

# 	plt.scatter(days[day_index], 
# 				x_vis[x_vis.columns.values[i]][day_index], 
# 				color='r') 

# 	plt.title(x_vis.columns.values[i]) 

# plt.show() 

# # SVM
# # importing necessary libraries
# from sklearn import datasets
# from sklearn.metrics import confusion_matrix
# from sklearn.model_selection import train_test_split

# # loading the iris dataset
# iris = datasets.load_iris()

# # X-> features, y-> features
# X = iris.data
# y= iris.target 

# # Dividing X, y into treain and test data
# X_test, X_train, y_test, y_train = train_test_split(X , y , random_state=0 );

# # Training a linear SVM classifier
# from sklearn.svm import SVC
# svm_model_linear = SVC(kernel=' linear', C=1).fit(X_train,y_train)
# svm_predictions = svm_model_linear.predict(X_test)

# # Model accuracy for X_test
# accuracy = svm_model_linear.score(X_test,y_test)

# # Creating a confusion matrix
# cm = confusion_matrix(y_test, svm_predictions)


# KNN(k-nearest neighbors) classifiers
# importing the necessary libraries

from sklearn import datasets
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split

# Loading iris dataset
iris = datasets.load_iris()

# X-> features , y-> features
X = iris.data
y= iris.target 

# Dividing X, y into train and test data

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state= 0 )

# Train the KNN classifier 
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors = 7).fit(X_train,y_train)

# Accuracy on X_test
accuracy = knn.score(X_test,y_test)
print("The accuracy of the KNN is:",accuracy * 100)


# Creating confusion matrix
knn_predictions = knn.predict(X_test)
cm = confusion_matrix(y_test, knn_predictions)


# Naive Bayes Classification algorithm 
# importing the necessary libraries
from sklearn import datasets
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split

# Loading the iris dataset
iris = datasets.load_iris()

# X-> features, y->features
X = iris.data
y = iris.target 
# Dividing X, y into training and test data
X_train, X_test, y_train, y_test = train_test_split(X, y , random_state = 0)

# Training a Naive Bayes Classifier 
from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB().fit(X_train,y_train)
gnb_predictions = gnb.predict(X_test)

# Accuracy on X_test
accuracy = gnb.score(X_test, y_test)
print("Accuray of the Naive Bayes Classifier is: ",accuracy)

# creating a confusion matrix
cm = confusion_matrix(y_test, gnb_predictions)


