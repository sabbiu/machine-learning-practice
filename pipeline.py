from sklearn import datasets
iris = datasets.load_iris()

X = iris.data
y = iris.target

from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = .5)


#one type of classifier:::decision tree
print('Decision Tree ->'),
from sklearn import tree
clf= tree.DecisionTreeClassifier()
clf.fit(X_train,y_train)

predictions = clf.predict(X_test)
#print(predictions)

#accuracy calculations
from sklearn.metrics import accuracy_score
print(accuracy_score(y_test, predictions)),
print()


#other type of classifier:::knn
print('KNN ->'),
from sklearn.neighbors import KNeighborsClassifier
my_clf = KNeighborsClassifier()
my_clf.fit(X_train,y_train)

my_predictions = my_clf.predict(X_test)
#print(predictions)

#accuracy calculations
print(accuracy_score(y_test, my_predictions))