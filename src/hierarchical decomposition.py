import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score, learning_curve
from sklearn.metrics import accuracy_score, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsOneClassifier

dataset_path = r"C:\Users\cauak\Downloads\Data.xlsx"
dados = pd.read_excel(dataset_path)

# Visualize as primeiras linhas do conjunto de dados
print(dados.head())

X = dados.drop('Class', axis=1)
y = dados['Class']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

hierarquia = {
    'Grupo_A': [1, 2, 3],
    'Grupo_B': [4, 5, 6],
    'Grupo_C': [7, 8],
    'Grupo_D': [9, 10]
}

modelos_por_grupo = {}

for grupo, classes in hierarquia.items():
    # Filtrar dados relevantes para o grupo atual
    X_grupo_train = X_train[y_train.isin(classes)]
    y_grupo_train = y_train[y_train.isin(classes)]

    # Verificar se há pelo menos uma amostra no grupo no conjunto de treinamento
    if not X_grupo_train.empty:
        # Criar e treinar o modelo para o grupo atual
        modelo = OneVsOneClassifier(LogisticRegression())
        modelo.fit(X_grupo_train, y_grupo_train)
        modelos_por_grupo[grupo] = modelo

y_pred_por_grupo = {}

for grupo, classes in hierarquia.items():
    # Filtrar dados relevantes para o grupo atual no conjunto de teste
    X_grupo_test = X_test[y_test.isin(classes)]

    # Fazer previsões se houver amostras no grupo
    if not X_grupo_test.empty:
        y_pred_grupo = modelos_por_grupo[grupo].predict(X_grupo_test)
        y_pred_por_grupo[grupo] = y_pred_grupo

y_pred = np.concatenate(list(y_pred_por_grupo.values())) if y_pred_por_grupo else np.array([])

# Avaliação usando k-fold cross validation
cross_val_scores = cross_val_score(LogisticRegression(), X, y, cv=5, scoring='accuracy', n_jobs=-1)
print(f'Acurácia média na validação cruzada: {np.mean(cross_val_scores)}')

# Avaliar a acurácia do modelo no conjunto de teste
accuracy = accuracy_score(y_test, y_pred)
print(f'Acurácia no conjunto de teste: {accuracy}')

# Exibir o relatório de classificação
print('Relatório de Classificação:')
print(classification_report(y_test, y_pred))

# Curvas de aprendizado
train_sizes, train_scores, test_scores = learning_curve(
    LogisticRegression(), X, y, cv=5, scoring='accuracy', n_jobs=-1
)

# Calcular as médias e desvios padrão
train_mean = np.mean(train_scores, axis=1)
train_std = np.std(train_scores, axis=1)
test_mean = np.mean(test_scores, axis=1)
test_std = np.std(test_scores, axis=1)

# Plotar as curvas de aprendizado
plt.figure(figsize=(10, 6))
plt.plot(train_sizes, train_mean, label='Treinamento')
plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.2)
plt.plot(train_sizes, test_mean, label='Teste')
plt.fill_between(train_sizes, test_mean - test_std, test_mean + test_std, alpha=0.2)
