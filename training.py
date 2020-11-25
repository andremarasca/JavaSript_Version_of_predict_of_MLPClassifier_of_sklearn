# https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPClassifier.html

from sklearn.neural_network import MLPClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
import pickle

X, y = make_classification(n_samples=100, random_state=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y,random_state=1)
clf = MLPClassifier(random_state=1, max_iter=300).fit(X_train, y_train)

clf.score(X_test, y_test)
clf.predict(X_test[:5, :])

pickle.dump(clf, open('clf.sav', 'wb'))
pickle.dump(X, open('X.sav', 'wb'))
pickle.dump(y, open('y.sav', 'wb'))