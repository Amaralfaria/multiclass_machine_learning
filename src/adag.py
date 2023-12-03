import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.multiclass import OneVsOneClassifier
import copy

def initialLevel(n_labels):
    first = 0
    second = (n_labels - 1) - n_labels%2
    level = []

    while(first < second):
        level.append(first)
        level.append(second)
        first+=1
        second-=1

    if n_labels%2 > 0:
        level.append(n_labels-1)

    return level



data_path = 'C:/Users/ionaa/OneDrive/Documentos/unb/quinto_semestre/iia/seminiario/data/Dry_Bean_Dataset.xlsx'
beans_data = pd.read_excel(data_path)

X = beans_data.iloc[:,1:-1]
y = beans_data.Class

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=16)

# scaler = StandardScaler()
# X_train = scaler.fit_transform(X_train)
# X_test = scaler.transform(X_test)

# class_array = np.unique(y)
# X_train_array = np.array(X_train)
# y_train_array = np.array(y_train)
# X_test_array = np.array(X_test)
# lables_data = []
# models = np.empty((len(class_array),len(class_array)), dtype='O')
# y_pred = np.empty(len(y_test), dtype='O')




# for i,label in enumerate(class_array):
#     data = np.array([row for index,row in enumerate(X_train_array) if y_train_array[index] == label])
#     lables_data.append(data)


# for i in range(len(class_array)):
#     for j in range(i+1,len(class_array)):
#         x_train_class = []
#         for row in lables_data[i]:
#             x_train_class.append(row)
#         for row in lables_data[j]:
#             x_train_class.append(row)


#         y_train_class = [1]*len(lables_data[i]) + [0]*len(lables_data[j])
#         model = LogisticRegression(max_iter=300,random_state=16)
#         # model = KNeighborsClassifier(n_neighbors=5)
#         model.fit(np.array(x_train_class),np.array(y_train_class))
#         models[i][j] = model
#         models[j][i] = -1



# for i,row in enumerate(X_test_array):
#     levelLabels = initialLevel(len(class_array))

#     while len(levelLabels) != 1:
#         next_level = []

#         for idx in range(0,len(levelLabels)-1 - (len(levelLabels)%2),2):
#             first = min(levelLabels[idx],levelLabels[idx+1])
#             second = max(levelLabels[idx],levelLabels[idx+1])

#             predicao = models[first][second].predict([row])
#             if predicao[0] == 1:
#                 next_level.append(first)
#             else:
#                 next_level.append(second)

#         if len(levelLabels)%2 > 0:
#             next_level.append(levelLabels[len(levelLabels)-1])



#         levelLabels = copy.deepcopy(next_level)
    

#     y_pred[i] = class_array[levelLabels[0]]

# for i,row in enumerate(X_test_array):
#     indexes = list(range(len(class_array)))

#     while len(indexes) != 1:
#         start = 0
#         end = (len(indexes) - 1) - len(indexes)%2
#         to_be_popped = []

#         while start < end:
#             first = min(indexes[start],indexes[end])
#             second = max(indexes[start],indexes[end])

#             predicao = models[first][second].predict([row])
#             if predicao[0] == 1:
#                 to_be_popped.append(second)
#             else:
#                 to_be_popped.append(first)

#             start += 1
#             end -= 1

#         for idx in to_be_popped:
#             indexes.remove(idx)
    

#     y_pred[i] = class_array[indexes[0]]


ovo = OneVsOneClassifier(LogisticRegression(max_iter=300))
ovo.fit(X_train,y_train)
y_pred = ovo.predict(X_test)



print(classification_report(y_test, y_pred))
print('Confusion matrix:')
print(confusion_matrix(y_test, y_pred))





    























    










