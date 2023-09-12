from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd

glass_data = pd.read_csv("glass.data")
glass_Data = glass_data.iloc[:, 1:10]
# type(glass_data)
# print(type(glass_data))
glass_name = glass_data.iloc[:, -1:]

print(glass_data)
print(glass_name)
print()
# #Impartire in Train si test



x_train, x_test, y_train, y_test = train_test_split(glass_Data, glass_name, test_size = 0.25)
print(x_train.shape)
print()
print(x_test.shape)

clf = MLPClassifier(hidden_layer_sizes=(5,5) , activation = 'relu', solver = 'adam',
                    max_iter = 2000, batch_size = 'auto', learning_rate='constant',
                    learning_rate_init = 0.01)
clf.fit(x_train, y_train.values.ravel())

pred = clf.predict(x_test)

print(pred)
print()

acc = accuracy_score(y_true= y_test , y_pred= pred)
print(acc)
print()
print(acc * 100)






