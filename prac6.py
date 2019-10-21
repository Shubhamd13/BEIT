import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split
from ann_visualizer.visualize import ann_viz
import os
os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz2.38/bin/'
#input
DiabetesData = np.loadtxt("NNdata.csv",delimiter=",")
print("Input data shape:",DiabetesData.shape)
DIP = np.delete(DiabetesData,8,1)
DOP = DiabetesData[:,8]
seed = 10
np.random.seed(seed)
(X_train, X_test, Y_train, Y_test) = train_test_split(DIP, DOP, test_size=0.30, random_state=seed)
#creating model
NNmodel = Sequential()
NNmodel.add(Dense(units=8, activation='sigmoid'))
NNmodel.add(Dense(units=6, activation='sigmoid'))
NNmodel.add(Dense(units=1, activation='sigmoid'))

#compiling model
NNmodel.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

NNmodel.fit(X_train,Y_train,nb_epoch=50, batch_size=20)

acc = NNmodel.evaluate(X_test, Y_test)
print("Accuracy: %.2f%%" % (acc[1]*100))

ann_viz(NNmodel,filename="viz.png", title="NN")
