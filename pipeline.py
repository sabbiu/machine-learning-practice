import random
from scipy.spatial import distance
def euc(a,b):
	return distance.euclidean(a,b)

class ScrappyKNN():
	def fit(self, X_train, y_train):
		self.X_train = X_train
		self.y_train = y_train

	#random prediction
	#def predict(self, X_test):
	#	predictions = []
	#	for row in X_test:
	#		label = random.choice(self.y_train)
	#		predictions.append(label)
#
#		return predictions

	def predict(self, X_test):
		predictions = []
		for row in X_test:
			label = self.closest(row)
			predictions.append(label)

		return predictions

	def closest(self, row):
		best_dist = euc(row, self.X_train[0])
		best_index = 0
		for i in range(1,len(self.X_train)):
			dist = euc(row, self.X_train[i])
			if dist < best_dist:
				best_dist = dist
				best_index = i

		return self.y_train[best_index]

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
print()


#our own classifier
print('KNN OUR->'),
our_clf = ScrappyKNN()
our_clf.fit(X_train,y_train)

our_predictions = our_clf.predict(X_test)
#print(predictions)

#accuracy calculations
print(accuracy_score(y_test, our_predictions))