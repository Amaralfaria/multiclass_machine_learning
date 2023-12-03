import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score, learning_curve
from sklearn.metrics import accuracy_score, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsOneClassifier

dataset_path = r"C:\Users\cauak\Downloads\Data.xlsx"
dataset = pd.read_excel(dataset_path)

X_custom = dataset.drop('Class', axis=1)
y_custom = dataset['Class']


X_train, X_test, y_train, y_test = train_test_split(X_custom, y_custom, test_size=0.2, random_state=42)


hierarquia = {
    'Grupo_A': [0, 1, 2],
    'Grupo_B': [3, 4, 5],
    'Grupo_C': [6, 7],
    'Grupo_D': [8, 9],
    'Grupo_E': [10, 11],
    'Grupo_F': [12, 13],
    'Grupo_G': [14, 15],
    'Grupo_H': [16, 17, 18, 19]
}


modelos_por_grupo = {}
y_pred_por_grupo = {}

for grupo, classes in hierarquia.items():  
    X_grupo_train = X_train[y_train.isin(classes)]
    y_grupo_train = y_train[y_train.isin(classes)]

    if not X_grupo_train.empty:
        modelo = OneVsOneClassifier(LogisticRegression())
        modelo.fit(X_grupo_train, y_grupo_train)
        modelos_por_grupo[grupo] = modelo

       
        X_grupo_test = X_test[y_test.isin(classes)]
        if not X_grupo_test.empty:
            y_pred_grupo = modelo.predict(X_grupo_test)
            y_pred_por_grupo[grupo] = y_pred_grupo


y_pred = np.concatenate(list(y_pred_por_grupo.values())) if y_pred_por_grupo else np.array([])


cross_val_scores = cross_val_score(LogisticRegression(), X_custom, y_custom, cv=5, scoring='accuracy', n_jobs=-1)
print(f'Acurácia média na validação cruzada: {np.mean(cross_val_scores)}')


accuracy = accuracy_score(y_test, y_pred)
print(f'Acurácia no conjunto de teste: {accuracy}')


print('Relatório de Classificação:')
print(classification_report(y_test, y_pred))


train_sizes, train_scores, test_scores = learning_curve(
    LogisticRegression(), X_custom, y_custom, cv=5, scoring='accuracy', n_jobs=-1
)


train_mean = np.mean(train_scores, axis=1)
train_std = np.std(train_scores, axis=1)
test_mean = np.mean(test_scores, axis=1)
test_std = np.std(test_scores, axis=1)


plt.figure(figsize=(10, 6))
plt.plot(train_sizes, train_mean, label='Treinamento')
plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.2)
plt.plot(train_sizes, test_mean, label='Teste')
plt.fill_between(train_sizes, test_mean - test_std, test_mean + test_std, alpha=0.2)
plt.xlabel('Tamanho do Conjunto de Treinamento')
plt.ylabel('Acurácia')
plt.title('Curva de Aprendizado')
plt.legend()
plt.show()