# -*- coding: utf-8 -*-
"""
Created on Sun Nov 17 21:23:45 2019

@author: Mauricio Rodríguez D
"""

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import warnings

#Llamamos desde la Libreria Sklearn 
#el Perceptron Multicapa Clasificador.
from sklearn.neural_network import MLPClassifier
#Llamamos desde la Libreria Sklearn 
#el Perceptron Multicapa de Regresión.
from sklearn.neural_network import MLPRegressor
#Llamamos desde la Libreria Sklearn 
#el proceso de división de entrenamiento y prueba.
from sklearn.model_selection import train_test_split
#Llamamos desde la Libreria Sklearn 
#el proceso de transformación de variables nominales a númericas.
from sklearn.preprocessing import LabelEncoder
#Llamamos desde la Libreria Sklearn 
#el proceso de transformación de variables corrigiendo su desviación estandar.
from sklearn.preprocessing import StandardScaler
from sklearn import metrics
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error
from sklearn.exceptions import ConvergenceWarning
from sklearn.model_selection import GridSearchCV

# Cargamos el conjunto de datos.
abalone=pd.read_csv('abalone2.csv')
#Información detallada de los datos
print(abalone.info())
"""
Observamos que tenemos un atributo
llamado Sexo con datos categóricos
Procedemos a cambiar este atributo
a númerico para facilitar su análisis
mediante la red Neuronal.
"""
encoder=LabelEncoder()
abalone['Sexo2']=encoder.fit_transform(abalone.Sexo.values)
#Mostramos si hubo el cambio de datos categóricos a numéricos
print(abalone.Sexo2.unique())


# Visualizando la distribución de las clases a través de un histograma.

#Histograma del atributo clase
ax=plt.subplots(1,1,figsize=(10,8))
sns.countplot('Anillos',data=abalone)
plt.title("Conteo de anillos en las instancias del dataset Abalone")
plt.show()

# Visualizando los histogramas de cada atributo.

#Histograma de atributos predictores

abalone.hist(edgecolor='black', linewidth=1.2)
fig=plt.gcf()
fig.set_size_inches(12,12)
plt.show()

# Diagrama de cajas de los atributos o variables independientes.

#boxplot de las variables numericas
abalone= abalone.drop('Sexo',axis=1)
box_data = abalone #variable representing the data array
box_target = abalone.Anillos #variable representing the labels array
sns.boxplot(data = box_data,width=0.5,fliersize=5)
sns.set(rc={'figure.figsize':(2,15)})
plt.show()

"""

Llamamos a dividir el dataset abalone
con la sentencia de entrenamiento y prueba de sklearn.
Para esto definimos con (X) las variables  de entrenamiento
y con (y) la variable de salida.

"""
#Definimos para en entrenamiento los atributos que consideramos más importantes
X= abalone[['Sexo2','Longitud','Diametro','Altura','Peso_entero','Peso_de_la_cascara']]
y = abalone['Anillos']
print (X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30)


#Procedemos a corregir la concurrencia de los datos con la desviación estandar
scaler = StandardScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

# Nuevo diagrama de cajas de los atributos o variables independientes.

#boxplot de las variables numericas
#abalone= abalone.drop('Sexo',axis=1)
box_data = abalone #variable representing the data array
box_target = abalone.Anillos #variable representing the labels array
sns.boxplot(data = box_data,width=0.5,fliersize=5)
sns.set(rc={'figure.figsize':(10,30)})
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

"""
#Entrenando un modelo de red neuronal MLP para clasificación
#MLPClassifier permite configurar las capas ocultas del modelo, 
la instrucción de abajo indica que el modelo tendrá
#dos capas ocultas cada una con 3 neuronas. Algo como esto 
hidden_layer_sizes = (3,3,2) indicarían tres capas ocultas 
con 3,3 y 2 neuronas respectivamente

"""
model =  MLPClassifier(hidden_layer_sizes = (3,3), alpha=0.01, max_iter=1500) 
model.fit(X_train, y_train) #Training the model

"""

Una vez entrenado el modelo, debemos evaluarlo sobre el conjunto de datos 
reservado para prueba, y utilizar algunas métricas para observar que tan bien 
quedo entrenado el modelo. En esta primera prueba utilizamos como métricas el 
porcentaje de precisión del modelo y la matriz de confusión.

"""

#Test the model
predictions = model.predict(X_test)
print(accuracy_score(y_test, predictions))
print("Confusion matrix:\n%s" % metrics.confusion_matrix(y_test, predictions))




# Ahora vamos a ajustar los parámetros del modelo utilizando GridSearch


#param_grid = [{'hidden_layer_sizes' : [(2,2), (3,3), (4,4), (5,4)], 'max_iter':[1100, 1500, 2000]}, 
              #{'alpha': [0.0001, 0.001, 0.01, 0.1]}]


#model = MLPClassifier()
#grid_search = GridSearchCV(model, param_grid, cv=3, scoring='accuracy', iid=False)
#grid_search.fit(X_train, y_train)


#print(grid_search.best_params_)




