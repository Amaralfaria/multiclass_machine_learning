import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.multiclass import OneVsOneClassifier
from sklearn.model_selection import train_test_split, cross_val_score, learning_curve
from sklearn.metrics import accuracy_score, classification_report
from sklearn.linear_model import LogisticRegression

# Carregue seus dados
dataset_path = 'data/Dry_Bean_Dataset.xlsx'
dataset = pd.read_excel(dataset_path)

X_custom = dataset.drop(['Class','Bean ID'], axis=1)
y_custom = dataset['Class']

# Divida os dados em treino e teste
X_train, X_test, y_train, y_test = train_test_split(X_custom, y_custom, test_size=0.2, random_state=42)

# Crie e treine o modelo
model = OneVsOneClassifier(LogisticRegression())


# Fazer previsões no conjunto de teste
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Avaliação usando k-fold cross validation
cross_val_scores = cross_val_score(model, X_custom, y_custom, cv=5, scoring='accuracy', n_jobs=-1)
print(f'Acurácia média na validação cruzada: {np.mean(cross_val_scores)}')

# Avaliar a acurácia do modelo no conjunto de teste
accuracy = accuracy_score(y_test, y_pred)
print(f'Acurácia no conjunto de teste: {accuracy}')

# Exibir o relatório de classificação
print('Relatório de Classificação:')
print(classification_report(y_test, y_pred))

# Curvas de aprendizado
train_sizes, train_scores, test_scores = learning_curve(
    model, X_custom, y_custom, cv=5, scoring='accuracy', n_jobs=-1
)

# Calcule as médias e desvios padrão
train_mean = np.mean(train_scores, axis=1)
train_std = np.std(train_scores, axis=1)
test_mean = np.mean(test_scores, axis=1)
test_std = np.std(test_scores, axis=1)

# Plote as curvas de aprendizado
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
