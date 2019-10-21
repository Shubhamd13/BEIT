import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
style.use("ggplot")
from sklearn import svm
from mlxtend.plotting import plot_decision_regions
#linear
X = np.array([[1,2],
             [5,8],
             [1.5,1.8],
             [8,8],
             [1,0.6],
             [9,11]])
#labels
y = np.array([0,1,0,1,0,1])
#watch plot
plt.scatter(X[:,0],X[:,1],c=y)
plt.show()
#define classfier with C = 1
clf = svm.SVC(kernel='linear', C = 1.0)
clf.fit(X,y)
plot_decision_regions(X=X, 
                      y=y,
                      clf=clf, 
                      legend=1)
plt.show()
print("Prediction of target for 0.58,0.76 values:")
t= clf.predict([[0.58,0.76]])
print(t)

print("Prediction of target for 10.58 ,10.76 value")
t= clf.predict([[10.58,10.76]])
print(t)

#rbf
X = np.array([[2,4],
             [2.3,4.2],
             [3,9],
             [3.3,9.3],
             [1,0.6],
             [1.2,0.7],
              [0,0],
              [4,16]])
#labels
y = np.array([0,1,0,1,0,1,0,1])
#watch plot
plt.scatter(X[:,0],X[:,1],c=y)
plt.show()
clf = svm.SVC(kernel='rbf',gamma="auto")
clf.fit(X,y)
plot_decision_regions(X=X, 
                      y=y,
                      clf=clf, 
                      legend=1)
plt.show()
#poly
#rbf
X = np.array([[2,4],
             [2.3,4.2],
             [3,9],
             [3.3,9.3],
             [1,0.6],
             [1.4,0.7],
              [0,0],
              [4,16]])
#labels
y = np.array([0,1,0,1,0,1,0,1])
poly_svc = svm.SVC(kernel='poly', degree=2, C=50).fit(X, y)
plot_decision_regions(X=X, 
                      y=y,
                      clf=poly_svc, 
                      legend=1)
plt.show()
