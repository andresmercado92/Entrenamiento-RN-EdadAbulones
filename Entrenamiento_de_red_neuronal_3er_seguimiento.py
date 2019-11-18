#!/usr/bin/env Pyhton3
# -*- coding: utf-8 -*-
"""
Created on Wen Nov 13 22:00:01 2019

@author: Mauricio Rodríguez Díaz
         Carmen Daly Vega Pérez
         Andrés Mercado Niño
         Paula Hernandez Vazquez
         
         
El conjunto de datos contiene una sola clase que es: Anillos. 
El conjunto de datos cuenta con 4177 instancias con 8 atributos.


         
"""
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import warnings

from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.neural_network import MLPRegressor
from sklearn import metrics
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error
from sklearn.exceptions import ConvergenceWarning
from sklearn.model_selection import GridSearchCV

#Cargando datos

abalone = pd.read_csv("abalone2.csv")
#Informacion de los datos
print(abalone.info())

#Histograma del atributo clase
ax=plt.subplots(1,1,figsize=(10,8))
sns.countplot('Anillos',data=abalone)
plt.title("Conteo de Anillos clasificados")
plt.show()


# Visualizando los histogramas de cada atributo.


#Histograma de atributos predictores

abalone.hist(edgecolor='black', linewidth=1.2)
fig=plt.gcf()
fig.set_size_inches(12,12)
plt.show()

# Diagrama de cajas de los atributos o variables independientes.


#boxplot de las variables numericas
abalone = abalone.drop('Sexo',axis=1)
box_data = abalone #variable representing the data array
box_target = abalone.Anillos #variable representing the labels array
sns.boxplot(data = box_data,width=0.5,fliersize=5)
#sns.set(rc={'figure.figsize':(5,10)})
plt.show()


# Observando la correlación entre variables permite descubrir posibles dependencias entre las variables independientes.


#observando correlacion entre variables
X = abalone.iloc[:, 0:9]
f, ax = plt.subplots(figsize=(10, 8))
corr = X.corr()
print(corr)
sns.heatmap(corr, mask=np.zeros_like(corr, dtype=np.bool), 
          cmap=sns.diverging_palette(220, 10, as_cmap=True),square=True, ax=ax, linewidths=.5)
plt.show() 

#print(abalone)

lista=list()

abalone["Class"]=""

for x in abalone["Anillos"]:
     if(x<11 ):  #Valores del 0-10
          lista.append("Clase1")

     if(x>10 and  x<21 ): #Valores del 11-20
          lista.append("Clase2")

     if(x>20): #Valores del 20 en adelante
          lista.append("Clase3")
print("Tamano lista: ", len(lista))

abalone["Class"] = lista

#print(abalone)

# Una vez observado y analizado las variables del conjunto de datos vamos a hacer una primera prueba preliminar para observar cómo se comportaría el modelo de red neuronal. La configuración de este primer modelo se indica a través de los parámetros de MPLClassifier


#Separando los datos en conjuntos de entrenaimiento y prueba
#X = abalone.iloc[:, :-1].values
#y = abalone.iloc[:, 4].values
X= abalone[['Longitud','Diametro','Altura','Peso_entero','Peso_entero_desvainado','Peso_de_las_visceras','Peso_de_la_cascara',]]
y = abalone['Class']
#print (X)
#print (y)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30)

#Como esta es una primera prueba prelimintar coloco esta instrucción para que nos me saque un warning
#debido a que el modelo no alcanza a converger
warnings.filterwarnings("ignore", category=ConvergenceWarning, module="sklearn")

#Entrenando un modelo de red neuronal MLP para clasificación
#MLPClassifier permite configurar las capas ocultas del modelo, la instrucción de abajo indica que el modelo tendrá
#dos capas ocultas cada una con 3 neuronas. Algo como esto hidden_layer_sizes = (3,3,2) indicarían tres capas ocultas con
#3,3 y 2 neuronas respectivamente
model =  MLPClassifier(hidden_layer_sizes = (3,3,2), alpha=0.01, max_iter=1000) 
model.fit(X_train, y_train) #Training the model


# Una vez entrenado el modelo, debemos evaluarlo sobre el conjunto de datos reservado para prueba, y utilizar algunas métricas para observar que tan bien quedo entrenado el modelo. En esta primera prueba utilizamos como métricas el porcentaje de precisión del modelo y la matriz de confusión.


#Test the model - Primer modelo
predictions = model.predict(X_test)
print("accuracy_score: ",accuracy_score(y_test, predictions))
print("Confusion matrix:\n%s" % metrics.confusion_matrix(y_test, predictions))


#Busqueda por cuadrícula (GridSearch)

#param_grid = [{'hidden_layer_sizes' : [(3,2,2), (3,3,4), (4,6,4), (5,5,4)], 'max_iter':[100, 1000, 2000],'alpha': [0.00001, 0.0001, 0.001, 00.1]}]

param_grid = [{'hidden_layer_sizes' : [(3,2,2), (2,2,2),(3,4,3),(3,2,2),(2,3,3),(6,10,1)], 
               'max_iter':[500,1000,2000], 
               'alpha': [0.1, 0.01,0.001,0.0001, 0.00001],
               'activation':['identity', 'relu'],
               'learning_rate_init':[0.001, 0.005, 0.01]
              }
             ]

model = MLPClassifier()
grid_search = GridSearchCV(model, param_grid, cv= 5, scoring='accuracy', iid=False)
grid_search.fit(X_train, y_train)
"""
results = pd.DataFrame(grid_search.cv_results_)
results.to_csv("results_training.csv", sep=",", index=False)
results.to_excel("results_training.xlsx", index=False)
"""
print(grid_search.best_params_)
"""
print(results)

print ("std_test_score: ",min(results["std_test_score"]))
BestParams= results[ results["std_test_score"] == min(results["std_test_score"]) ]
print ("std_train_score: ",BestParams["std_train_score"])

print(BestParams)

"""
"""
#Mejor modelo
model1 =  MLPClassifier(learning_rate_init= 0.01, activation= 'relu', hidden_layer_sizes = (3, 4, 3), alpha=0.001, max_iter=2000) 
model1.fit(X_train, y_train) #Training the model

predictions1 = model1.predict(X_test)
print("accuracy_score: ",accuracy_score(y_test, predictions1))
print("Confusion matrix:\n%s" % metrics.confusion_matrix(y_test, predictions1))

"""



















