# supervised ml example
from sklearn import tree

# Define the features and labels
features = [[140, 1], [130, 1], [150, 0], [170, 0]]
labels = [0, 0, 1, 1]

# Initialize and train the classifier
clf = tree.DecisionTreeClassifier()
clf = clf.fit(features, labels)

# Make a prediction
print(clf.predict([[160, 0]]))

