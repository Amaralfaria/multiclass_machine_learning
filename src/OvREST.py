import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, recall_score

def generateMetrics(algoritmos,y_test):
    acuracias = {}
    f1_scores = {}
    recalls = {}


    # Avalie cada modelo
    for nome, algoritmo in algoritmos:
        # Treine o modelo
        ovr = OneVsRestClassifier(algoritmo())
        ovr.fit(X_train, y_train)
        y_pred = ovr.predict(X_test)

        # Calcule as métricas
        acuracia = accuracy_score(y_test, y_pred)
        f1_multiclasse = f1_score(y_test, y_pred, average='weighted')
        recall_multiclasse = recall_score(y_test, y_pred, average='weighted')

        # Armazene as métricas nos dicionários
        acuracias[nome] = acuracia
        f1_scores[nome] = f1_multiclasse
        recalls[nome] = recall_multiclasse

    return acuracias,f1_scores,recalls

def plotar_metricas(algoritmos,y_test,X_test_array,class_array):
    metricas = ['Acurácia', 'F1-score', 'Recall']
    acuracias, f1_scores, recalls = generateMetrics(algoritmos,y_test,X_test_array,class_array)


    cores = ['blue', 'orange', 'green','black']

    # Crie uma figura para cada métrica

    for i, metrica in enumerate(metricas):
        valores_metrica = [acuracias, f1_scores, recalls][i]
        
        plt.figure()
        barras = plt.bar(valores_metrica.keys(), valores_metrica.values(), color=cores)

        for barra in barras:
            altura = barra.get_height()
            plt.text(barra.get_x() + barra.get_width() / 2, altura + 0.02, f'{altura:.3f}', ha='center', va='bottom')


        plt.ylim(0, 1)
        plt.title(f'{metrica} - Comparação entre Modelos')
        plt.ylabel(metrica)

    plt.show()

data_path = 'data/Dry_Bean_Dataset.xlsx'


beans_data = pd.read_excel(data_path)
X = beans_data.iloc[:,1:-1]
y = beans_data.Class


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=16)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Minha implementação
# models = []

# for label in np.unique(y):
#     filter = lambda r:(1 if r == label else -1)
#     y_train_class = np.array([filter(row) for row in y_train])
#     y_test_class = np.array([filter(row) for row in y_test])

#     model = DecisionTreeClassifier()
#     model.fit(X_train,y_train_class)
#     models.append((model,label))


# y_pred = np.empty(len(y_test), dtype='O')
# max_prob = np.zeros(len(y_train))

# for model in models:
#     probability = model[0].predict_proba(X_test)[:,1]
#     for i,prob in enumerate(probability):
#         if max_prob[i] <= prob:
#             max_prob[i] = prob
#             y_pred[i] = model[1]


#Usando biblioteca

# model = DecisionTreeClassifier()

# ovr = OneVsRestClassifier(model)

# ovr.fit(X_train,y_train)

# y_pred = ovr.predict(X_test)
# print(np.unique(y_pred))


# print(classification_report(y_test, y_pred))
# print('Confusion matrix:')
# print(confusion_matrix(y_test, y_pred))

modelos = [
    ('Decision Tree', DecisionTreeClassifier()),
    ('KNN', KNeighborsClassifier(n_neighbors=3)),
    ('Logistic Regression', LogisticRegression(max_iter=300)),
    ('Naive Bayes', GaussianNB())
]

# Lista para armazenar as acurácias de cada modelo
acuracias = []

# Treine e avalie cada modelo
for nome, modelo in modelos:
    ovr = OneVsRestClassifier(modelo)
    ovr.fit(X_train, y_train)
    y_pred = ovr.predict(X_test)
    acuracia = accuracy_score(y_test, y_pred)
    acuracias.append((nome, acuracia))

# Crie um gráfico de barras para comparar as acurácias
nomes_modelos, valores_acuracia = zip(*acuracias)
barras = plt.bar(nomes_modelos, valores_acuracia, color=['blue', 'green', 'red', 'black'])  # Adicione mais cores conforme necessário

for barra in barras:
        altura = barra.get_height()
        plt.text(barra.get_x() + barra.get_width() / 2, altura + 0.02, f'{altura:.3f}', ha='center', va='bottom')
plt.ylabel('Acurácia')
plt.title('Comparação de Acurácia entre Modelos')
plt.ylim(0, 1)  # Defina o limite y de 0 a 1 para a acurácia
plt.show()


