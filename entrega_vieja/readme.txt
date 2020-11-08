Laboratorio 1 IPLN Grupo 18

El proyecto consiste en un pipeline de tres pasos que se encarga del preprocesamiento de las entradas para luego pasarlas al modelo de clasificación. Los primeros dos pasos son el preprocesado mientras que el último es el modelo a utilizar.

El primer paso, al que llamamos “cleaner”, se encarga de remover todos los espacios en los extremos de las entradas (“trimming”) y las convierte a minúsculas.

El segundo paso, denominado “vectorizer”, tiene múltiples pasos internos.
Primero, se toma cada entrada y se la pasa por el parser en español de la librería Spacy.
A continuación, con la entrada parseada, se recorre cada uno de los tokens y se los sustituye por su lema (evitando los pronombres). 
Por último, se filtran todas las palabras vacías de la entrada.

Una vez realizado todo el preprocesado se crean los vectores de “bag of words”. 
El equipo encontró que la solución que arroja los mejores resultados está formada por vectores de “bag of words” formados por unigramas y bigramas.

Con los vectores generados, a partir del conjunto de entrenamiento, se entrenaron varios modelos pertenecientes a la librería scikit-learn. El modelo que obtuvo mejores resultados fue un modelo de regresión logística.
Para este, los hiperparámetros por defecto fue los que llevaron a obtener mejores resultados en el conjunto de prueba.

Versiones:

Nltk - 3.4.5
Scikit-learn - 0.21.3
Spacy - 2.2.1
Pandas - 0.25.2
Python - 3.7.4

