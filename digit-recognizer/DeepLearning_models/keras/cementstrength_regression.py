import numpy as np
import pandas as pd
import keras
import tensorflow as tf

from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense


input_data = pd.read_csv("https://s3-api.us-geo.objectstorage.softlayer.net/cf-courses-data/CognitiveClass/DL0101EN/labs/data/concrete_data.csv")
input_data.head()
input_data.describe()
input_data.isnull().sum()

input_data_columns = input_data.columns
x_data = input_data[input_data_columns[input_data_columns != 'Strength']]
y_data = input_data['Strength']
n_columns = x_data.shape[1]
# print(x_data.head)


# data_norm = (x_data - x_data.mean)/ x_data.std()


# print(y_data.head)

X_train, x_test, Y_train, y_test = train_test_split(x_data, y_data, test_size=0.3, random_state=42)


mse_list = []
def regression_model() -> keras.Model:
      model = Sequential()
      model.add(Dense(10, activation='relu', input_shape=(n_columns, )))
      model.add(Dense(1))
      model.compile(optimizer='adam', loss="mean_squared_error",  metrics=['mean_squared_error'])
      return model


for i in range(50):
      model = regression_model()
      model.fit(X_train, Y_train, epochs=50)
      predicted_y = model.predict(x_test)
      mse = tf.keras.metrics.mean_squared_error(y_test, predicted_y)
      mse_list.append(mse)

for i in range(50):
      print("For i = {} : mean is {}, std is {}".format(i, np.mean(mse_list[i]), np.std(mse_list[i])))

# print(np.mean(mse_list))
# print(np.std(mse_list))
# fit the model
# model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=10, batch_size=200, verbose=2)

# # evaluate the model
# scores = model.evaluate(x_test, y_test, verbose=0)
# print("mse: {} \n Error: {}".format(scores[1], 100-scores[1]*100))