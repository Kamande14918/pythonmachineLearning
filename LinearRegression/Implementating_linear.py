# Simple linear regression implementation
# import numpy as np
# import matplotlib.pyplot as plt

# def estimate_coef(x, y):
#     # Number of observations/points
#     n = np.size(x)
    
#     # Mean of x and y vector
#     m_x, m_y = np.mean(x), np.mean(y)
    
#     # Calculating cross-deviation and deviation about x
#     SS_xy = np.sum(y*x) - n*m_y*m_x
#     SS_xx = np.sum(x*x) - n*m_x*m_x
    
#     # Calculating regression coefficients
#     b_1 = SS_xy / SS_xx
#     b_0 = m_y - b_1*m_x
    
#     return (b_0, b_1)

# def predict(x, b):
#     return b[0] + b[1]*x

# def main():
#     # Observations data
#     x = np.array([0,1,2,3,4,5,6,7,8,9])
#     y = np.array([1,3,2,5,7,8,8,9,10,12])
    
#     # Estimating coefficients
#     b = estimate_coef(x, y)
#     print(f"Estimated coefficients:\nb_0={b[0]} \nb_1={b[1]}")
    
#     # Predicted response vector
#     y_pred = predict(x, b)
    
#     # Plotting the actual points as a scatter plot
#     plt.scatter(x, y, color="b", marker="o", s=30)
    
#     # Plotting the regression line
#     plt.plot(x, y_pred, color="g")
    
#     # Putting labels
#     plt.xlabel('x')
#     plt.ylabel('y')
    
#     # Function to show plot
#     plt.show()

# if __name__ == "__main__":
#     main()


# # Multiple Linear Regression Implementation
# from sklearn.model_selection import train_test_split
# import matplotlib.pyplot as plt
# import numpy as np
# from sklearn import datasets, linear_model, metrics
# import pandas as pd
# # Loading the Boston Housing Dataset

# data_url = "http://lib.stat.cmu.edu/datasets/boston"
# raw_df = pd.read_csv(data_url, sep="\s+",skiprows=22, header=None)


# # Processing dataset

# X = np.hstack([raw_df.values[::2, :], raw_df.values[1::2, :2]])
# y = raw_df.values[1::2, 2]

# # Splitting Data into Training and Testing Sets
# X_train, X_test, \
#     y_train, y_test = train_test_split(X,y, test_size = 0.4, random_state =1)
# # Creating and training Linear Regression Model 

# reg = linear_model.LinearRegression()
# reg.fit(X_train,y_train)

# # Evaluating model performance 
# # printing Regression Coefficients
# print("Coefficients:", reg.coef_)

# # Variance score: 1 means perfect prediction 
# print("Variance score: {}".format(reg.score(X_test,y_test)))

# #plotting residual errors in training data
# # plot for residual error'

# # Setting plotting style
# plt.style.use('fivethirtyeight')

# #  plotting residual error in training data 
# plt.scatter(reg.predict(X_train),
# reg.predict(X_train) - y_train,
# color = "green", s=10, label = 'Train data')

# # Plotting residual errors in test data set
# plt.scatter(reg.predict(X_test),
#             reg.predict(X_test)- y_test,
#             color="blue", s=10,
#             label='Test data')
# # Plotting line for zero residual error
# plt.hlines(y=0, xmin=0, xmax=50, linewidth=2)

# # plotting legend
# plt.legend(loc='upper right')

# # Plot title 
# plt.title("Residual errors")

# # method call for showing the plot
# plt.show()



# Implementing polynomial Regression using Python

