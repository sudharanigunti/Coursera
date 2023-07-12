import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC

data = pd.read_csv("C:\\Users\\sgunti\\Downloads\\digit-recognizer\\train.csv")
x = data.iloc[:, 1:]
y = data.iloc[:, 0]

test_data = pd.read_csv(".\\test.csv")
x_final_test = test_data.iloc[:, 1:]
y_final_test = test_data.iloc[:, 0]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
# print(np.shape(x_train))
# print(np.shape(x_test))
# print(x_train.head(1))

svc_model = SVC(kernel = "poly")
model = svc_model.fit(X=x_train, y=y_train)
y_predict = model.predict(x_test)
print(accuracy_score(y_test, y_predict))

y_test = y_test.to_numpy()
count = 0

for i in range(len(y_test)):
    if y_test[i] != y_predict[i]:
        count = count+1

differ_elements = count
total_elements = len(y_test)
accuracy = (total_elements - differ_elements) / total_elements

print(accuracy)

model = svc_model.fit(X=x_train, y=y_train)

y_final_predict = model.predict(x_final_test)
print(accuracy_score(y_final_test, y_final_predict))
