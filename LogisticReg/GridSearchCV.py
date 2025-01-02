# import pandas as pd
# import numpy as np
# from sklearn.metrics import classification_report, confusion_matrix
# from sklearn.datasets import load_breast_cancer
# from sklearn.svm import SVC

# cancer = load_breast_cancer()

# # The dataset is presented in a dictionary form:
# # print(cancer.keys())

# df_feat = pd.DataFrame(cancer['data'], 
# 					columns = cancer['feature_names']) 

# # cancer column is our target 
# df_target = pd.DataFrame(cancer['target'], 
# 					columns =['Cancer']) 

# print("Feature Variables: ") 
# print(df_feat.info()) 
# # print("Dataframe looks like: ")
# # print(df_feat.head())


# # Train test split
# from sklearn.model_selection import train_test_split

# X_train, X_test, y_train, y_test = train_test_split(df_feat,np.ravel(df_target), test_size = 0.30, random_state = 101)

# # train the model on train set
# model = SVC()
# model.fit(X_train,y_train)

# # Print prediction results
# predictions = model.predict(X_test)
# print(classification_report(y_test, predictions))

# # Usse GridSearchCV to find the best parameters
# from sklearn.model_selection import GridSearchCV

# # Defining the parameter range
# param_grid ={'C':[0.1, 1, 10, 100, 1000],
#              'gamma':[1, 0.1, 0.01, 0.001, 0.0001],
#              'kernel':['rbf']}
# grid = GridSearchCV(SVC(),param_grid, refit =True, verbose = 3)

# # Fitting the model for grid search
# grid.fit(X_train, y_train)

# # print the best parameter after tuning
# print(grid.best_params_)


# # Print how our model looks after hyper-parameter tuning
# print(grid.best_estimator_)

# grid_predictions = grid.predict(X_test)

# # Print classification reeport 
# print(classification_report(y_test, grid_predictions))


# importing libraries 
import numpy as np 
import matplotlib.pyplot as plt 
from sklearn.datasets import make_circles 
from mpl_toolkits.mplot3d import Axes3D 

# generating data 
X, Y = make_circles(n_samples = 500, noise = 0.02) 

# visualizing data 
plt.scatter(X[:, 0], X[:, 1], c = Y, marker = '.') 
plt.show() 
# adding a new dimension to X 
X1 = X[:, 0].reshape((-1, 1)) 
X2 = X[:, 1].reshape((-1, 1)) 
X3 = (X1**2 + X2**2) 
X = np.hstack((X, X3)) 

# visualizing data in higher dimension 
fig = plt.figure() 
axes = fig.add_subplot(111, projection = '3d') 
axes.scatter(X1, X2, X1**2 + X2**2, c = Y, depthshade = True) 
plt.show() 

# create support vector classifier using a linear kernel 
from sklearn import svm 

svc = svm.SVC(kernel = 'linear') 
svc.fit(X, Y) 
w = svc.coef_ 
b = svc.intercept_ 

# plotting the separating hyperplane 
x1 = X[:, 0].reshape((-1, 1)) 
x2 = X[:, 1].reshape((-1, 1)) 
x1, x2 = np.meshgrid(x1, x2) 
x3 = -(w[0][0]*x1 + w[0][1]*x2 + b) / w[0][2] 

fig = plt.figure() 
axes2 = fig.add_subplot(111, projection = '3d') 
axes2.scatter(X1, X2, X1**2 + X2**2, c = Y, depthshade = True) 
axes1 = fig.gca() 
axes1.plot_surface(x1, x2, x3, alpha = 0.01) 
plt.show() 
