import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import Perceptron
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, f1_score, recall_score
import copy

def generateMetrics(algoritmos,y_test,X_test_array,class_array):
    acuracias = {}
    f1_scores = {}
    recalls = {}


    # Avalie cada modelo
    for nome, algoritmo in algoritmos:
        # Treine o modelo
        models = createModels(class_array, algoritmo)
        
        # Faça previsões no conjunto de teste
        y_pred = predict(X_test_array,class_array,models)

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


def createModels(class_array, model):

    models = np.empty((len(class_array),len(class_array)), dtype='O')
    
    for i in range(len(class_array)):
        for j in range(i+1,len(class_array)):
            x_train_class = []
            for row in lables_data[i]:
                x_train_class.append(row)
            for row in lables_data[j]:
                x_train_class.append(row)


            y_train_class = [1]*len(lables_data[i]) + [0]*len(lables_data[j])
            model = copy.deepcopy(model)
            # model = ''
            # if class_algoritmo.__name__ == 'LogisticRegression':
            #     model = class_algoritmo(max_iter=300)
            # else:
            #     model = class_algoritmo()
            model.fit(np.array(x_train_class),np.array(y_train_class))
            models[i][j] = model
            models[j][i] = -1

    return models

def predict(X_test_array, class_array, models):
    y_pred = np.empty(len(X_test_array), dtype='O')

    for i,row in enumerate(X_test_array):
        levelLabels = initialLevel(len(class_array))

        while len(levelLabels) != 1:
            next_level = []

            for idx in range(0,len(levelLabels)-1 - (len(levelLabels)%2),2):
                first = min(levelLabels[idx],levelLabels[idx+1])
                second = max(levelLabels[idx],levelLabels[idx+1])

                predicao = models[first][second].predict([row])
                if predicao[0] == 1:
                    next_level.append(first)
                else:
                    next_level.append(second)

            if len(levelLabels)%2 > 0:
                next_level.append(levelLabels[len(levelLabels)-1])



            levelLabels = copy.deepcopy(next_level)

        y_pred[i] = class_array[levelLabels[0]]

    return y_pred
    



data_path = 'C:/Users/ionaa/OneDrive/Documentos/unb/quinto_semestre/iia/seminiario/data/Dry_Bean_Dataset.xlsx'
beans_data = pd.read_excel(data_path)

X = beans_data.iloc[:,1:-1]
y = beans_data.Class

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=16)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

class_array = np.unique(y)
X_train_array = np.array(X_train)
y_train_array = np.array(y_train)
X_test_array = np.array(X_test)
lables_data = []


for i,label in enumerate(class_array):
    data = np.array([row for index,row in enumerate(X_train_array) if y_train_array[index] == label])
    lables_data.append(data)



# models = createModels(class_array)
# y_pred = predict(X_test_array,class_array,models)

algoritmos = [
    ('Decision Tree', DecisionTreeClassifier()),
    ('KNN', KNeighborsClassifier(n_neighbors=13)),
    ('Logistic Regression', LogisticRegression()),
    ('Naive Bayes', GaussianNB())
]



# plotar_metricas(algoritmos,y_test,X_test_array,class_array)
acuracias = []


for nome, algoritmo in algoritmos:
    models = createModels(class_array,algoritmo)
    y_pred = predict(X_test_array,class_array,models)
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



# print(classification_report(y_test, y_pred))
# print('Confusion matrix:')
# print(confusion_matrix(y_test, y_pred))





    























    










