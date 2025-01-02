# # importing the necessary libraries
# from  sklearn.datasets import load_breast_cancer
# from sklearn.linear_model import LogisticRegression
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import accuracy_score

# # Load the breast cancer dataset
# X, y = load_breast_cancer(return_X_y = True)

# # Split the train and test dataset
# X_train, X_test, \
#     y_train, y_test = train_test_split(X,y, test_size=0.20,
#                                        random_state=23)
#     # LogisticRegression
# clf = LogisticRegression(random_state = 0)
# clf.fit(X_train,y_train)
# # Prediction
# y_pred = clf.predict(X_test)

# acc = accuracy_score(y_test,y_pred)
# print("Logistic Regression model accuracy (in %):", acc * 100)


# Multinomial Logistic Regression

from sklearn.model_selection import train_test_split
from sklearn import datasets, linear_model, metrics

# Load digit datasets
digits = datasets.load_digits()

# Defining the feature matrix(X) and response vector(y)
X= digits.data
y= digits.target

# Splitting X and y into training and testing sets

X_train,X_test ,\
    y_train, y_test =train_test_split(X,y,
                                      test_size=0.4,
                                      random_state=1)

# Create logistic regression object
reg = linear_model.LogisticRegression()
    
    
# Train the model using the training set 
reg.fit(X_train,y_train)

# Making predictions on the testing dataset 
y_pred = reg.predict(X_test)

#Comparing actual response values (y_test)
# With predicted response values (y_pred)
print("Logistic Regression model accuracy (in %):",
      metrics.accuracy_score(y_test,y_pred)* 100)
    