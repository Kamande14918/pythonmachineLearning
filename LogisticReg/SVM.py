# Importing scikit learn with make_blobs
from sklearn.datasets import make_blobs
import numpy as np
import matplotlib.pyplot as plt

# Creating datasets X containing n_samples
# Y containing two classes
X, Y = make_blobs(n_samples=5000, centers=2, random_state=0, cluster_std=0.40)

# Plotting scatters
plt.scatter(X[:, 0], X[:, 1], c=Y, s=50, cmap='spring')

# Creating a line space between -1 to 3.5
xfit = np.linspace(-1, 3.5)

# Plotting the line between the different sets of data
for m, b, d in [(1, 0.65, 0.33), (0.5, 1.6, 0.55), (-0.2, 2.9, 0.2)]:
    yfit = m * xfit + b
    plt.plot(xfit, yfit, '-k')
    plt.fill_between(xfit, yfit - d, yfit + d, edgecolor='none', color='#AAAAAA', alpha=0.4)

plt.xlim(-1, 3.5)
# plt.show()

# importing required libraries
import pandas as pd 

# Reading the csv file and extracting class column to y
x = pd.read_csv("C:\...\cancer.csv")
a = np.array(x)
y = a[:,30] # class having 0 and 1

# Extracting two features
x = np.column_stack(x.malignant, x.benign)

# 569 samples and 2 features
x.shape

print(x),(y)

# import support vector classifier
# "Support Vector Classifier"

from sklearn.svm import SVC
clf = SVC(kernel='linear')

# Fitting x samples and y classes
clf.fit(x,y)
# Predict new values
clf.predict([[120,990]])
clf.predict([[85,550]])
