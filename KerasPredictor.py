import numpy as np
import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
import matplotlib.pyplot as plt


df = pd.read_csv("Data/winequality-white.csv", sep=";")

# Divide data in training and testing

dfTrain, dfTest = train_test_split(df, test_size=0.2)

# Divide target from data

X = np.array(dfTrain.ix[:,:11])
Y = np.array(dfTrain.ix[:,11])

xTest = np.array(dfTest.ix[:,:11])
yTest = np.array(dfTest.ix[:,11])

# Scale data

scaler1 = MinMaxScaler()
scaledX = scaler1.fit_transform(X)

scaler2 = MinMaxScaler()
scaledTest = scaler2.fit_transform(xTest)

# More Preprocessing should go here

# Define seed
seed = 3
np.random.seed(seed)

# Define and compile model

model = Sequential()
model.add(Dense(11, input_dim=11, kernel_initializer="normal", activation="relu"))
model.add(Dense(1, kernel_initializer="normal"))

adam = Adam(lr=0.01)

model.compile(loss='mean_absolute_percentage_error', optimizer=adam)

# Fit the model

model.fit(scaledX, Y, batch_size=scaledX.shape[0], epochs=500, validation_data=(scaledTest, yTest))

trainPrediction = model.predict(scaledX)
testPrediction = model.predict(scaledTest)

f = open('results.txt', 'w')

for i in range(len(yTest)):
	f.write(str(int(round(testPrediction[i][0])))+" "+str(yTest[i])+"\n")

f.close()