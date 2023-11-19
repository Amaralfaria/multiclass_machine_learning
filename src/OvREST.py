import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

data_path = 'C:/Users/ionaa/OneDrive/Documentos/unb/quinto_semestre/iia/seminiario/data/Dry_Bean_Dataset.xlsx'


beans_data = pd.read_excel(data_path)
X = beans_data.iloc[:,1:-1]
y = beans_data.Class

# Assuming X_train, y_train are your training data and labels
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=16)


models = []

for label in np.unique(y):
    filter = lambda r:(1 if r == label else -1)
    y_train_class = np.array([filter(row) for row in y_train])
    y_test_class = np.array([filter(row) for row in y_test])

    model = LogisticRegression(max_iter=300, random_state=16)
    model.fit(X_train,y_train_class)
    models.append((model,label))


y_pred = np.empty(len(y_test), dtype='O')
max_prob = np.zeros(len(y_train))

for model in models:
    probability = model[0].predict_proba(X_test)[:,1]
    for i,prob in enumerate(probability):
        if max_prob[i] < prob:
            max_prob[i] = prob
            y_pred[i] = model[1]



print(classification_report(y_test, y_pred))
print('Confusion matrix:')
print(confusion_matrix(y_test, y_pred))


