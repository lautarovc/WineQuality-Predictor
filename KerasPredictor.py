import sys
import numpy as np
import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
import matplotlib.pyplot as plt

from numpy.random import seed
from tensorflow import set_random_seed

# Usage

def printUsage():
	print("Usage: KerasPredictor.py white|red")
	sys.exit(2)

try:
	wine = sys.argv[1]
	if wine != "red" and wine != "white":
		printUsage()
except:
	printUsage()

# Load data

df = pd.read_csv("Data/winequality-"+wine+".csv", sep=";")

# Preprocessing should go here


# Divide data in training and testing

dfTrain, dfTest = train_test_split(df, test_size=0.2, shuffle=False)

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


# Define seed

seed(1)
set_random_seed(1)


# Define and compile model

model = Sequential()

if wine=="white":
	model.add(Dense(11, input_dim=11,  activation="relu"))
	model.add(Dense(15, activation="relu"))
	model.add(Dense(8, activation="relu"))

else:
	model.add(Dense(10, input_dim=11,  activation="relu"))
	model.add(Dense(8, activation="relu"))


model.add(Dense(1))

adam = Adam(lr=0.01)

model.compile(loss='mean_absolute_percentage_error', optimizer=adam)

# Fit the model

model.fit(scaledX, Y, batch_size=scaledX.shape[0], epochs=410, validation_data=(scaledTest, yTest))

trainPrediction = model.predict(scaledX)
testPrediction = model.predict(scaledTest)

f = open('results-'+wine+'.txt', 'w')

for i in range(len(yTest)):
	f.write(str(int(round(testPrediction[i][0])))+" "+str(yTest[i])+"\n")

f.close()