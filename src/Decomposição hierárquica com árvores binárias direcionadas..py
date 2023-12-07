import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score, learning_curve
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt

# Carrega os dados
dataset_path = r"C:\Users\cauak\Downloads\Data.xlsx"
dataset = pd.read_excel(dataset_path)

# Remove colunas desnecessárias
dataset = dataset.drop("Bean ID", axis=1)

# Nome da coluna do atributo alvo
coluna_atributo_alvo = 'Class'

# Use LabelEncoder para converter rótulos de classe para valores numéricos
label_encoder = LabelEncoder()
dataset[coluna_atributo_alvo] = label_encoder.fit_transform(dataset[coluna_atributo_alvo])

# Selecionar colunas numéricas para normalização
cols_to_normalize = dataset.select_dtypes(include=['number']).columns.difference([coluna_atributo_alvo])
dataset[cols_to_normalize] = dataset[cols_to_normalize].fillna(dataset[cols_to_normalize].mean())

# Normalizar os dados
scaler = MinMaxScaler()
dataset[cols_to_normalize] = scaler.fit_transform(dataset[cols_to_normalize])

# Separar atributo base e features
X = dataset.drop(coluna_atributo_alvo, axis=1)
y = dataset[coluna_atributo_alvo]

# Dividir os dados em treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Definir uma classe para representar um nó na árvore hierárquica
class BinaryTreeNode:
    def __init__(self, depth):
        self.depth = depth
        self.model = None
        self.left_child = None
        self.right_child = None

# Função para treinar modelos em cada nível da árvore hierárquica
def train_binary_tree_models(node, X, y, max_depth):
    # Divide os dados em dois ramos
    mask = X[X.columns[node.depth % len(X.columns)]] <= X[X.columns[node.depth % len(X.columns)]].median()
    X_left = X[mask]
    y_left = y[mask]
    X_right = X[~mask]
    y_right = y[~mask]

    # Treina modelos para os ramos esquerdo e direito
    node.model = DecisionTreeClassifier()
    node.model.fit(X, y)

    if node.depth == max_depth:
        return

    # Cria os filhos (nós para os ramos esquerdo e direito)
    node.left_child = BinaryTreeNode(node.depth + 1)
    node.right_child = BinaryTreeNode(node.depth + 1)

    # Chama recursivamente para os próximos níveis
    train_binary_tree_models(node.left_child, X_left, y_left, max_depth)
    train_binary_tree_models(node.right_child, X_right, y_right, max_depth)

# Função para fazer previsões usando a árvore hierárquica
def predict_with_binary_tree(node, X, max_depth):
    if node.model is None:
        return np.zeros(len(X))  # Se o modelo não está treinado para este nó, retorna um array de zeros

    if node.depth == max_depth:
        return node.model.predict(X)

    # Divide os dados em dois ramos
    mask = X[X.columns[node.depth % len(X.columns)]] <= X[X.columns[node.depth % len(X.columns)]].median()
    X_left = X[mask]
    X_right = X[~mask]

    # Chama recursivamente para os ramos esquerdo e direito
    predictions_left = predict_with_binary_tree(node.left_child, X_left, max_depth)
    predictions_right = predict_with_binary_tree(node.right_child, X_right, max_depth)

    # Combina as previsões dos ramos
    predictions = np.zeros(len(X))  
    predictions[mask] = predictions_left
    predictions[~mask] = predictions_right
    return predictions

# Treina modelos em cada nível da árvore hierárquica
max_depth = 2  # Número máximo de níveis da árvore
root_node = BinaryTreeNode(0)
train_binary_tree_models(root_node, X_train, y_train, max_depth=max_depth)

# Faz previsões no conjunto de teste usando a árvore treinada
y_pred = predict_with_binary_tree(root_node, X_test, max_depth)

# Calcula e imprime a acurácia no conjunto de teste
accuracy = accuracy_score(y_test, y_pred)
print(f'Acurácia no conjunto de teste: {accuracy}')

# Adiciona k-fold cross-validation
kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cross_val_scores = cross_val_score(root_node.model, X, y, cv=kfold, scoring='accuracy', n_jobs=-1)

# Relatório de classificação
classification_rep = classification_report(y_test, y_pred)
print(f'Relatório de Classificação:\n{classification_rep}')

# Preenche os resultados na tabela
cv_results = {
    'Modelo': [],
    'Acurácia Média': [],
    'Desvio Padrão Acurácia': []
}
cv_results['Modelo'].append('Árvore Hierárquica')
cv_results['Acurácia Média'].append(np.mean(cross_val_scores))
cv_results['Desvio Padrão Acurácia'].append(np.std(cross_val_scores))

# Adiciona curvas de aprendizado
train_sizes, train_scores, test_scores = learning_curve(
    root_node.model, X, y, cv=kfold, scoring='accuracy', n_jobs=-1
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
plt.title('Curva de Aprendizado para Árvore Hierárquica')
plt.legend()
plt.show()
