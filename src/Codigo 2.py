import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score, learning_curve
from sklearn.metrics import accuracy_score, classification_report, precision_score, recall_score, f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.multiclass import OneVsOneClassifier
from sklearn.preprocessing import MinMaxScaler

# Carrega os dados
dataset_path = r"C:\Users\cauak\Downloads\Data.xlsx"
dataset = pd.read_excel(dataset_path)

# Converta a coluna 'Class' para valores numéricos usando LabelEncoder
# Remove colunas desnecessárias
dataset = dataset.drop("Bean ID", axis=1)

# Tratamento de dados categóricos (se necessário)
# dataset = pd.get_dummies(dataset, columns=['sua_coluna_categorica'])

# Tratamento de valores ausentes (se necessário)
# dataset = dataset.dropna()  # ou use técnicas de imputação

# Nome da coluna do atributo alvo
coluna_atributo_alvo = 'Class'

# Seleciona colunas numéricas para normalização
cols_to_normalize = dataset.select_dtypes(include=['number']).columns.difference([coluna_atributo_alvo])

# Normaliza os dados
scaler = MinMaxScaler()
dataset[cols_to_normalize] = scaler.fit_transform(dataset[cols_to_normalize])

# Separar atributo base e features
X = dataset.drop(coluna_atributo_alvo, axis=1)
y = dataset[coluna_atributo_alvo]

# Divide os dados em treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Cria e treina os modelos
models = {
    'Logistic Regression': OneVsOneClassifier(LogisticRegression()),
    'Naive Bayes': OneVsOneClassifier(GaussianNB()),
    'K Neighbors': OneVsOneClassifier(KNeighborsClassifier()),
    'Decision Tree': OneVsOneClassifier(DecisionTreeClassifier())
}

# Armazena acurácias médias na validação cruzada
cv_accuracies = {}

# Avalia e compara modelos
for model_name, model in models.items():
    # Faz previsões no conjunto de teste
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # Avaliação do modelo
    accuracy = accuracy_score(y_test, y_pred)
    print(f'Acurácia no conjunto de teste para {model_name}: {accuracy}')

    # Outras métricas de avaliação
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')

    print(f'Precision: {precision}, Recall: {recall}, F1-score: {f1}')

    # Adiciona k-fold cross-validation
    kfold_accuracies = cross_val_score(model, X, y, cv=5, scoring='accuracy', n_jobs=-1)
    cv_accuracy = np.mean(kfold_accuracies)
    cv_accuracies[model_name] = cv_accuracy

    print(f'Acurácia média na validação cruzada para {model_name}: {cv_accuracy}')

    # Adiciona curvas de aprendizado
    train_sizes, train_scores, test_scores = learning_curve(
        model, X, y, cv=5, scoring='accuracy', n_jobs=-1
    )

    # Calcula as médias e desvios padrão
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    test_mean = np.mean(test_scores, axis=1)
    test_std = np.std(test_scores, axis=1)

    # Plota as curvas de aprendizado
    plt.figure(figsize=(10, 6))
    plt.plot(train_sizes, train_mean, label='Treinamento')
    plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.2)
    plt.plot(train_sizes, test_mean, label='Teste')
    plt.fill_between(train_sizes, test_mean - test_std, test_mean + test_std, alpha=0.2)
    plt.xlabel('Tamanho do Conjunto de Treinamento')
    plt.ylabel('Acurácia')
    plt.title(f'Curva de Aprendizado para {model_name}')
    plt.legend()
    plt.show()

# Cria uma tabela comparativa das acurácias médias na validação cruzada
comparison_df = pd.DataFrame(list(cv_accuracies.items()), columns=['Modelo', 'Acurácia Média na Validação Cruzada'])

# Cria gráfico de barras com cores diferentes
colors = ['skyblue', 'lightcoral', 'lightgreen', 'orange']

plt.figure(figsize=(10, 6))
plt.bar(comparison_df['Modelo'], comparison_df['Acurácia Média na Validação Cruzada'], color=colors)
plt.xlabel('Modelo')
plt.ylabel('Acurácia Média na Validação Cruzada')
plt.title('Comparação de Acurácias na Validação Cruzada')
plt.xticks(rotation=45, ha='right')

# Adiciona valores nas barras
for i, value in enumerate(comparison_df['Acurácia Média na Validação Cruzada']):
    plt.text(i, value + 0.01, f'{value:.3f}', ha='center', va='bottom')

# Exibe o gráfico de barras
plt.tight_layout()
plt.show()
