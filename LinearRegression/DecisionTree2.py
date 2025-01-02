import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, plot_tree, export_text
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

# Load dataset
iris = load_iris()
X, y = iris.data, iris.target

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train the decision tree classifier
clf_gini = DecisionTreeClassifier(criterion='gini', random_state=42)
clf_gini.fit(X_train, y_train)

clf_entropy = DecisionTreeClassifier(criterion='entropy', random_state=42)
clf_entropy.fit(X_train, y_train)

# Visualize the decision tree using plot_tree
plt.figure(figsize=(20,10))
plot_tree(clf_gini, feature_names=iris.feature_names, class_names=iris.target_names, filled=True)
plt.title("Decision Tree using Gini Index")
plt.show()

plt.figure(figsize=(20,10))
plot_tree(clf_entropy, feature_names=iris.feature_names, class_names=iris.target_names, filled=True)
plt.title("Decision Tree using Entropy")
plt.show()

# Optionally, get a textual representation of the tree
tree_text_gini = export_text(clf_gini, feature_names=iris.feature_names)
print("Decision Tree using Gini Index:\n", tree_text_gini)

tree_text_entropy = export_text(clf_entropy, feature_names=iris.feature_names)
print("Decision Tree using Entropy:\n", tree_text_entropy)