"""
This code is based from Machine Learning A-Z™:
Hands-On Python & R In Data Science
Section 16, Lecture 112
Naive Bayes
"""

# importing libraries
import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
import pandas as pd

# importing dataset
dataset = pd.read_csv('dataset.txt')
X = dataset.iloc[:, :-1].values     # get first two columns
y = dataset.iloc[:, 2].values       # get last column
print("\nX values:\n", X)
print("\ny values:\n", y)
# encoding categorical data
from sklearn.preprocessing import LabelEncoder
labelencoder_y = LabelEncoder()
y = labelencoder_y.fit_transform(y)
print("\ny values after using labelencoder:\n", y)

# splitting dataset into Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)
print("\nX_train values:\n", X_train)
print("\nX_test values:\n", X_test)
print("\ny_train values:\n", y_train)
print("\ny_test values:\n", y_test)

# feature scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
print("\nX_train values after feature scaling:\n", X_train)
print("\nX_test values after feature scaling:\n", X_test)

# fitting Training set to the classifier
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(X_train, y_train)

# predicting the Test set
y_pred = classifier.predict(X_test)
print("\ny_pred values:\n", y_pred)

# making confusion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print("\nconfusion matrix values:\n", cm)

# Training set visualization
from matplotlib.colors import ListedColormap
X_set, y_set = X_train, y_train
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'blue')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('red', 'blue'))(i), label = j)
plt.title('Naive Bayes (Test set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()

# additional sample prediction (hard-coded)
predict_this = sc.transform([[30, 25485]])
print("\nsample prediction:\n", classifier.predict(predict_this))