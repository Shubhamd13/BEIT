import numpy as np
import matplotlib.pyplot as plt
#from matplotlib import style
#style.use("ggplot")
from sklearn import svm
from mlxtend.plotting import plot_decision_regions
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

iris_dataset = datasets.load_iris()
X = iris_dataset.data[:, :2]
y = iris_dataset.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
#watch plot
plt.scatter(X[:,0],X[:,1],c=y)
plt.show()
#define classfier with C = 1
clf = svm.SVC(kernel='linear', C = 1.0)
clf.fit(X_train,y_train)
plot_decision_regions(X=X,y=y,clf=clf,legend=1)
plt.show()
y_pred=clf.predict(X_test)

print(accuracy_score(y_test, y_pred))

#rbf
clf = svm.SVC(kernel='rbf')
clf.fit(X_train,y_train)
plot_decision_regions(X=X, 
                      y=y,
                      clf=clf, 
                      legend=1)
plt.show()
y_pred=clf.predict(X_test)
print(accuracy_score(y_test, y_pred))
#poly
poly_svc = svm.SVC(kernel='poly').fit(X_train, y_train)
plot_decision_regions(X=X, 
                      y=y,
                      clf=poly_svc, 
                      legend=1)
plt.show()

y_pred=poly_svc.predict(X_test)

print(accuracy_score(y_test, y_pred))
