from numpy import array
from keras.models import Sequential
from keras.layers import Dense
import pandas as pd


# load the dataset
dataset=pd.read_csv('Data.csv')

# split data set  into input (X) and output (y) variables
X = dataset.iloc[:,0:3].values
Y = dataset.iloc[:,3].values

# define the keras model
model = Sequential()
#ANN mode with input layer,hidden layer, output layer
model.add(Dense(6, input_dim=3, activation='relu'))
model.add(Dense(12, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

# compile the keras model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# fit the keras model on the dataset
model.fit(X, Y, epochs=100, batch_size=50)

# evaluate the keras model
_, accuracy = model.evaluate(X, Y)
print('Accuracy: %.2f' % (accuracy*100))

# save model to h5 file ( model and weights)
model.save("CompleteModel222.h5")
print("saved model to disk")

"""
# save  model to JSON
model_json = model.to_json()
with open ("model.json","w") as json_file:
    json_file.write(model_json)
# save weights to HDF5
model.save_weights("model.h5")
print("saved model to disk")
"""









