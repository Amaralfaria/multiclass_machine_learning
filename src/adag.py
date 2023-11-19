import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split


data_path = 'C:/Users/X03152467150/Documents/unb/inteligencia_artificial/seminario/src/beans.xlsx'
beans_data = pd.read_excel(data_path)

X = beans_data.iloc[:,1:-1]
y = beans_data.Class

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=16)

class_list = np.unique(y)
X_train_array = np.array(X_train)
y_train_array = np.array(y_train)
lables_data = []
models = np.empty((len(class_list),len(class_list)), dtype='O')


for i,label in enumerate(class_list):
    data = np.array([row for index,row in enumerate(X_train_array) if y_train_array[index] == label])
    lables_data.append(data)


for i in range(len(class_list)):
    for j in range(i+1,len(class_list)):
        x_train_class = []
        for row in lables_data[i]:
            x_train_class.append(row)
        for row in lables_data[j]:
            x_train_class.append(row)


        y_train_class = [1]*len(lables_data[i]) + [0]*len(lables_data[j])
        model = LogisticRegression(max_iter=300,random_state=16)
        model.fit(np.array(x_train_class),np.array(y_train_class))
        models[i][j] = model
        models[j][i] = model










    










