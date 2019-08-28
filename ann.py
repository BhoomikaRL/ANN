
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 

#************************Data Preprocessing*************************************#

dataset=pd.read_csv('Churn_Modelling.csv')

x=dataset.iloc[:,3:13].values
y=dataset.iloc[:,13].values

from sklearn.preprocessing import LabelEncoder
lEncoderCol1=LabelEncoder()
lEncoderCol2=LabelEncoder()

x[:,1]=lEncoderCol1.fit_transform(x[:,1])
x[:,2]=lEncoderCol2.fit_transform(x[:,2])

from sklearn.preprocessing import OneHotEncoder

ohEncoder=OneHotEncoder(categorical_features=[1])

x=ohEncoder.fit_transform(x).toarray()

x=x[:,1:]

from sklearn.preprocessing import StandardScaler
sScaler=StandardScaler()
x=sScaler.fit_transform(x)

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,random_state=0,test_size=0.2)


#**********************Data Preprocessing Ends*********************************#

import keras
from keras.models import Sequential
from keras.layers import Dense

annClassifier=Sequential()

annClassifier.add(Dense(input_dim=11,activation='relu',unit=6))

annClassifier.add(Dense(activation='relu',unit=6))

annClassifier.add(Dense(activation='relu',unit=6))

annClassifier.add(Dense(unit=1,activation='sigmoid'))

annClassifier.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])

history=annClassifier.fit(x_train,x_test,batch_size=10,)

plt.subplot(1,2,1)
plt.plot(history.history['acc'])
plt.subplot(1,2,2)
plt.plot(history.history['loss'])

adam=keras.optimizers.adam()

