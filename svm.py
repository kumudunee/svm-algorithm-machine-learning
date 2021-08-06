import pandas as pd
from sklearn.datasets import load_iris
from matplotlib import pyplot as plt
from sklearn.model_selection import  train_test_split
from sklearn.svm import SVC


iris = load_iris()

print(iris.keys())

print(iris.feature_names)

df = pd.DataFrame(iris.data, columns=iris.feature_names)
print(df.head)
df['target'] = iris.target
print(df)

print(iris.target_names)

print(df[df.target == 2].head())

print(df[df.target == 1].head())
df['flower_name'] = df.target.apply(lambda x: iris.target_names[x])
print(df)

df0 = df[df.target==0]
df1 = df[df.target==1]
df2 = df[df.target==2]
print(df0.head)
print(df1.head)
print(df2.head)

plt.xlabel('sepal length (cm)')
plt.ylabel('sepal width (cm)')
plt.scatter(df0['sepal length (cm)'],df0['sepal width (cm)'],color='green',marker='+')
plt.scatter(df1['sepal length (cm)'],df1['sepal width (cm)'],color='blue',marker='.')
plt.show()

x=df.drop(['target','flower_name'],axis='columns')
print(x)
y=df.target
print(y)

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2)
print(len(x_train))

print(len(x_test))

model = SVC(C=10)
#model = SVC(gamma=1)

model.fit(x_train, y_train)
print(model)
print(model.score(x_test,y_test))
#print(model)
