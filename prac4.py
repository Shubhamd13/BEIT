import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from matplotlib import pyplot as plt
df=pd.DataFrame({
    'x':[12, 28, 28, 18, 29, 30, 74, 45, 45, 52, 51, 52, 55, 53, 55, 61, 64, 69, 72],
    'y':[39, 13, 30, 52, 84, 46, 55, 89, 63, 70, 66, 63, 58, 23, 14, 8, 19, 7, 24]
})
print(df)
plt.scatter(df['x'],df['y'])
plt.show()

k_rng=range(1,10)
sse=[]
#print(sse.shape)
for k in k_rng:
    km=KMeans(n_clusters=k)
    km.fit(df[['x','y']])
    f=km.inertia_
    sse=np.append(sse,f)
    print(km.inertia_)
plt.plot(k_rng,sse)
plt.show()




km=KMeans(n_clusters=4)
predicted=km.fit_predict(df[['x','y']])
print(predicted)

df['cluster']=predicted
print(df)


df0=df[df.cluster==0]
df1=df[df.cluster==1]
df2=df[df.cluster==2]
df3=df[df.cluster==3]
#fetch all the rows where cluster value =0,1,2
plt.scatter(df0.x,df0.y,color='green')
plt.scatter(df1.x,df1.y,color='red')
plt.scatter(df2.x,df2.y,color='blue')
plt.scatter(df3.x,df3.y,color='yellow')
plt.xlabel('x')
plt.ylabel('y')
print("the final centroids are")
print(km.cluster_centers_)
plt.scatter(km.cluster_centers_[:,0],km.cluster_centers_[:,1],color='black',marker='*')
plt.show()