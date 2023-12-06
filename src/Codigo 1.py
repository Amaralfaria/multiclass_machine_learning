import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score, learning_curve
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, classification_report

# Carrega os dados
dataset_path = r"C:\Users\cauak\Downloads\Data.xlsx"
dataset = pd.read_excel(dataset_path)

# Remove colunas desnecessárias
dataset = dataset.drop("Bean ID", axis=1)

# Nome da coluna do atributo alvo
coluna_atributo_alvo = 'Class'

# Seleciona colunas numéricas para normalização
cols_to_normalize = dataset.select_dtypes(include=['number']).columns.difference([coluna_atributo_alvo])

# Normaliza os dados
scaler = MinMaxScaler()
dataset[cols_to_normalize] = scaler.fit_transform(dataset[cols_to_normalize])

# Separa atributo alvo do restante
X = dataset.drop(coluna_atributo_alvo, axis=1)
y = dataset[coluna_atributo_alvo]

# Divide os dados em treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Cria e treina modelos
models = {
    'Decision Tree': OneVsRestClassifier(DecisionTreeClassifier()),
    'Naive Bayes': OneVsRestClassifier(GaussianNB()),
    'K Neighbors': OneVsRestClassifier(KNeighborsClassifier()),
    'Logistic Regression': OneVsRestClassifier(LogisticRegression())
}

# Armazena acurácias médias na validação cruzada
cv_results = {'Modelo': [], 'Acurácia Média': [], 'Desvio Padrão Acurácia': []}

# Avalia e compara modelos
for model_name, model in models.items():
    # Faz previsões no conjunto de teste
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # Avaliação do modelo
    accuracy = accuracy_score(y_test, y_pred)
    print(f'Acurácia no conjunto de teste para {model_name}: {accuracy}')

    # Relatório de classificação
    print(f'Relatório de Classificação para {model_name}:')
    print(classification_report(y_test, y_pred))

    # Adiciona k-fold cross-validation
    kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cross_val_scores = cross_val_score(model, X, y, cv=kfold, scoring='accuracy', n_jobs=-1)
    
    # Preenche os resultados na tabela
    cv_results['Modelo'].append(model_name)
    cv_results['Acurácia Média'].append(np.mean(cross_val_scores))
    cv_results['Desvio Padrão Acurácia'].append(np.std(cross_val_scores))

    # Adiciona curvas de aprendizado
    train_sizes, train_scores, test_scores = learning_curve(
        model, X, y, cv=kfold, scoring='accuracy', n_jobs=-1
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

# Cria uma tabela comparativa detalhada das acurácias na validação cruzada
comparison_df = pd.DataFrame(cv_results)

# Ordena o DataFrame pela coluna 'Acurácia Média' em ordem decrescente
comparison_df = comparison_df.sort_values(by='Acurácia Média', ascending=False)

# Plota um gráfico de barras vertical
plt.figure(figsize=(10, 6))
bar_plot = sns.barplot(x='Modelo', y='Acurácia Média', hue='Modelo', data=comparison_df, palette='viridis', dodge=False, legend=False)

# Adiciona os valores de acurácia acima das barras
for index, value in enumerate(comparison_df['Acurácia Média']):
    bar_plot.text(index, value, f'{value:.3f}', ha='center', va='bottom', fontsize=8)

# Configurações adicionais para melhor visualização
plt.ylim(0, 1.0)  # Define o limite y de 0 a 1.0
plt.axhline(y=1.0, color='red', linestyle='--', linewidth=0.8)
plt.xticks(rotation=45)  # Rotaciona os rótulos para melhor legibilidade
plt.xlabel('Modelo')
plt.ylabel('Acurácia Média na Validação Cruzada')
plt.title('Comparação de Acurácias na Validação Cruzada')
plt.tight_layout()

# Salva o gráfico como uma imagem
plt.savefig('comparacao_acuracias_modificado.png')

# Exibe o gráfico
plt.show()
