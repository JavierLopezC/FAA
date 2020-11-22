# -*- coding: utf-8 -*-

# coding: utf-8
import sklearn
from Datos import Datos
from sklearn import preprocessing
from sklearn.compose import ColumnTransformer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.metrics import accuracy_score
import numpy as np
from Clasificador import Clasificador, ClasificadorNaiveBayes, ClasificadorVecinosProximos, ClasificadorRegresionLogistica
from EstrategiaParticionado import ValidacionSimple, ValidacionCruzada
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.naive_bayes import GaussianNB, MultinomialNB
import matplotlib.pyplot as	 plt

from EstrategiaParticionado import EstrategiaParticionado

def repr_grafica(nombre, xlabel, ylabel, num_lines, line_labels, colors, x, y):
	plt.figure()
	lw = 2
	for i in range(0, num_lines):
		plt.plot(x[i], y[i], color=colors[i], lw=lw, label=line_labels[i])

	plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
	plt.xlim([0.0, 1.0])
	plt.ylim([0.0, 1.05])
	plt.xlabel(xlabel)
	plt.ylabel(ylabel)
	plt.title(nombre)
	plt.legend(loc="lower right")
	plt.show()



def valores_roc(data, pred):
	fp = 0
	fn = 0
	tp = 0
	tn = 0
	for i in range(0, len(pred)):
		if pred[i] == data[i]:
			if pred[i] == 1:
				tp += 1
			elif pred[i] == 0:
				tn += 1
		elif pred[i] != data[i]:
			if pred[i] == 1:
				fp += 1
			elif pred[i] == 0:
				fn += 1
	return tp, tn, fp, fn

def main():
	diabetes = Datos("pima-indians-diabetes.data")
	wdbc = Datos("wdbc.data")
	
	print("Diabetes:\n")
	print("nominalAtributos:")
	print(diabetes.nominalAtributos)
	print("\nDiccionario:")
	print(diabetes.diccionario)
	print("\nDatos:")
	print(diabetes.datos)

	print("\nValidacion Cruzada NB")
	print("\nValidando con clasificador propio:")
	nb = ClasificadorNaiveBayes()
	vc = ValidacionCruzada()
	error = Clasificador.validacion(vc, diabetes, nb)
	print("\n Error medio:")
	print("sin Laplace: " + str(error[0]))
	print("con Laplace: " + str(error[1]))


	#######################################################################################################################

	print("\nValidacion Cruzada K-NN")
	print("\nValidando con clasificador propio:")
	knn = ClasificadorVecinosProximos()
	vc = ValidacionCruzada()
	error = Clasificador.validacion(vc, diabetes, knn)
	print("\nError medio k=1")
	print("euclidea: " + str(error[0]))
	print("manhattan: " + str(error[1]))
	print("mahalanobis: " + str(error[2]))
	print("\nError medio k=5")
	print("euclidea: " + str(error[3]))
	print("manhattan: " + str(error[4]))
	print("mahalanobis: " + str(error[5]))
	print("\nError medio k=11")
	print("euclidea: " + str(error[6]))
	print("manhattan: " + str(error[7]))
	print("mahalanobis: " + str(error[8]))
	print("\nError medio k=21")
	print("euclidea: " + str(error[9]))
	print("manhattan: " + str(error[10]))
	print("mahalanobis: " + str(error[11]))

	#######################################################################################################################

	print("\nValidacion Cruzada Regresión Logística")
	print("\nValidando con clasificador propio:")
	reg = ClasificadorRegresionLogistica()
	vc = ValidacionCruzada()
	error = Clasificador.validacion(vc, diabetes, reg)
	print("\n Error medio: " + str(error))

	#######################################################################################################################

	print("\n\nWdbc:\n")
	print("nominalAtributos:")
	print(wdbc.nominalAtributos)
	print("\nDiccionario:")
	print(wdbc.diccionario)
	print("\nDatos:")
	print(wdbc.datos)

	print("\n\nWdbc:")

	print("\nValidacion Cruzada NB")
	print("\nValidando con clasificador propio:")
	nb = ClasificadorNaiveBayes()
	vc = ValidacionCruzada()
	error = Clasificador.validacion(vc, wdbc, nb)
	print("\n Error medio:")
	print("sin Laplace: " + str(error[0]))
	print("con Laplace: " + str(error[1]))

	#######################################################################################################################

	print("\nValidacion Cruzada K-NN")
	print("\nValidando con clasificador propio:")
	knn = ClasificadorVecinosProximos()
	vc = ValidacionCruzada()
	error = Clasificador.validacion(vc, wdbc, knn)
	print("\nError medio k=1")
	print("euclidea: " + str(error[0]))
	print("manhattan: " + str(error[1]))
	print("mahalanobis: " + str(error[2]))
	print("\nError medio k=5")
	print("euclidea: " + str(error[3]))
	print("manhattan: " + str(error[4]))
	print("mahalanobis: " + str(error[5]))
	print("\nError medio k=11")
	print("euclidea: " + str(error[6]))
	print("manhattan: " + str(error[7]))
	print("mahalanobis: " + str(error[8]))
	print("\nError medio k=21")
	print("euclidea: " + str(error[9]))
	print("manhattan: " + str(error[10]))
	print("mahalanobis: " + str(error[11]))

	#######################################################################################################################

	print("\nValidacion Cruzada Regresión Logística")
	print("\nValidando con clasificador propio:")
	reg = ClasificadorRegresionLogistica()
	vc = ValidacionCruzada()
	error = Clasificador.validacion(vc, wdbc, reg)
	print("\nError medio: " + str(error))

########################################################################################################################
########################################################################################################################

	# SKLEARN

	X = diabetes.datos[:, :-1]
	Y = diabetes.datos[:, -1]
	x = np.transpose(X)

	X2 = wdbc.datos[:, :-1]
	Y2 = wdbc.datos[:, -1]
	x2 = np.transpose(X2)
	
	print("\n\nDiabetes:")
	
	####################################################################################################################
	print("************Knn SKLEARN************\n")
	print("\nK = 1\n")
	clf = KNeighborsClassifier(n_neighbors=1, p=2, metric='euclidean')
	score = cross_val_score(clf, X, Y, cv=10, n_jobs=-1)
	error_media_sk = 1 - score.mean()
	error_std_sk = score.std()
	print("Error medio sklearn euclidean: " + str(error_media_sk))
	print("Desviación media del error sklearn euclidean: " + str(error_std_sk))

	clfM = KNeighborsClassifier(n_neighbors=1, p=2, metric='manhattan')
	scoreM = cross_val_score(clfM, X, Y, cv=10, n_jobs=-1)
	error_media_skM = 1 - scoreM.mean()
	error_std_skM = scoreM.std()
	print("Error medio sklearn manhattan: " + str(error_media_skM))
	print("Desviación media del error sklearn manhattan: " + str(error_std_skM))

	clfMahalan = KNeighborsClassifier(n_neighbors=1, p=2, metric='mahalanobis', metric_params={'V': np.cov(x)})
	scoreMahalan = cross_val_score(clfMahalan, X, Y, cv=10, n_jobs=-1)
	error_media_skMahalan = 1 - scoreMahalan.mean()
	error_std_skMahalan = scoreMahalan.std()
	print("Error medio sklearn mahalanobis: " + str(error_media_skMahalan))
	print("Desviación media del error sklearn mahalanobis: " + str(error_std_skMahalan))

	print("\nK = 3\n")
	clf3 = KNeighborsClassifier(n_neighbors=3, p=2, metric='euclidean')
	score3 = cross_val_score(clf3, X, Y, cv=10, n_jobs=-1)
	error_media_sk3 = 1 - score3.mean()
	error_std_sk3 = score3.std()
	print("Error medio sklearn euclidean: " + str(error_media_sk3))
	print("Desviación media del error sklearn euclidean: " + str(error_std_sk3))

	clf3M = KNeighborsClassifier(n_neighbors=3, p=2, metric='manhattan')
	score3M = cross_val_score(clf3M, X, Y, cv=10, n_jobs=-1)
	error_media_sk3M = 1 - score3M.mean()
	error_std_sk3M = score3M.std()
	print("Error medio sklearn manhattan: " + str(error_media_sk3M))
	print("Desviación media del error sklearn manhattan: " + str(error_std_sk3M))

	clf3Mahalan = KNeighborsClassifier(n_neighbors=3, p=2, metric='mahalanobis', metric_params={'V':np.cov(x)})
	score3Mahalan = cross_val_score(clf3Mahalan, X, Y, cv=10, n_jobs=-1)
	error_media_sk3Mahalan = 1 - score3Mahalan.mean()
	error_std_sk3Mahalan = score3Mahalan.std()
	print("Error medio sklearn mahalanobis: " + str(error_media_sk3Mahalan))
	print("Desviación media del error sklearn mahalanobis: " + str(error_std_sk3Mahalan))


	print("\nK = 5\n")
	clf5 = KNeighborsClassifier(n_neighbors=5, p=2, metric='euclidean')
	score5 = cross_val_score(clf5, X, Y, cv=10, n_jobs=-1)
	error_media_sk5 = 1 - score5.mean()
	error_std_sk5 = score5.std()
	print("Error medio sklearn euclidean: " + str(error_media_sk5))
	print("Desviación media del error sklearn euclidean: " + str(error_std_sk5))

	clf5M = KNeighborsClassifier(n_neighbors=5, p=2, metric='manhattan')
	score5M = cross_val_score(clf5M, X, Y, cv=10, n_jobs=-1)
	error_media_sk5M = 1 - score5M.mean()
	error_std_sk5M = score5M.std()
	print("Error medio sklearn manhattan: " + str(error_media_sk5M))
	print("Desviación media del error sklearn manhattan: " + str(error_std_sk5M))

	clf5Mahalan = KNeighborsClassifier(n_neighbors=5, p=2, metric='mahalanobis', metric_params={'V': np.cov(x)})
	score5Mahalan = cross_val_score(clf5Mahalan, X, Y, cv=10, n_jobs=-1)
	error_media_sk5Mahalan = 1 - score5Mahalan.mean()
	error_std_sk5Mahalan = score5Mahalan.std()
	print("Error medio sklearn mahalanobis: " + str(error_media_sk5Mahalan))
	print("Desviación media del error sklearn mahalanobis: " + str(error_std_sk5Mahalan))

	print("\nK = 11\n")
	clf11 = KNeighborsClassifier(n_neighbors=11, p=2, metric='euclidean')
	score11 = cross_val_score(clf11, X, Y, cv=10, n_jobs=-1)
	error_media_sk11 = 1 - score11.mean()
	error_std_sk11 = score11.std()
	print("Error medio sklearn euclidean: " + str(error_media_sk11))
	print("Desviación media del error sklearn euclidean: " + str(error_std_sk11))

	clf11M = KNeighborsClassifier(n_neighbors=11, p=2, metric='manhattan')
	score11M = cross_val_score(clf11M, X, Y, cv=10, n_jobs=-1)
	error_media_sk11M = 1 - score11M.mean()
	error_std_sk11M = score11M.std()
	print("Error medio sklearn manhattan: " + str(error_media_sk11M))
	print("Desviación media del error sklearn manhattan: " + str(error_std_sk11M))

	clf11Mahalan = KNeighborsClassifier(n_neighbors=11, p=2, metric='mahalanobis', metric_params={'V': np.cov(x)})
	score11Mahalan = cross_val_score(clf11Mahalan, X, Y, cv=10, n_jobs=-1)
	error_media_sk11Mahalan = 1 - score11Mahalan.mean()
	error_std_sk11Mahalan = score11Mahalan.std()
	print("Error medio sklearn mahalanobis: " + str(error_media_sk11Mahalan))
	print("Desviación media del error sklearn mahalanobis: " + str(error_std_sk11Mahalan))

	print("\nK = 21\n")
	clf21 = KNeighborsClassifier(n_neighbors=21, p=2, metric='euclidean')
	score21 = cross_val_score(clf21, X, Y, cv=10, n_jobs=-1)
	error_media_sk21 = 1 - score21.mean()
	error_std_sk21 = score21.std()
	print("Error medio sklearn euclidean: " + str(error_media_sk21))
	print("Desviación media del error sklearn euclidean: " + str(error_std_sk21))

	clf21M = KNeighborsClassifier(n_neighbors=21, p=2, metric='manhattan')
	score21M = cross_val_score(clf21M, X, Y, cv=10, n_jobs=-1)
	error_media_sk21M = 1 - score21M.mean()
	error_std_sk21M = score21M.std()
	print("Error medio sklearn manhattan: " + str(error_media_sk21M))
	print("Desviación media del error sklearn manhattan: " + str(error_std_sk21M))

	clf21Mahalan = KNeighborsClassifier(n_neighbors=21, p=2, metric='mahalanobis', metric_params={'V': np.cov(x)})
	score21Mahalan = cross_val_score(clf21Mahalan, X, Y, cv=10, n_jobs=-1)
	error_media_sk21Mahalan = 1 - score21Mahalan.mean()
	error_std_sk21Mahalan = score21Mahalan.std()
	print("Error medio sklearn mahalanobis: " + str(error_media_sk21Mahalan))
	print("Desviación media del error sklearn mahalanobis: " + str(error_std_sk21Mahalan))
	
	####################################################################################################################

	print("\n*************** Regresión Logística ***************")
	print("\nLogisticRegression")
	clfRL = LogisticRegression(max_iter=10000, fit_intercept=1)
	scoreRL = cross_val_score(clfRL, X, Y, cv=10, n_jobs=-1)
	error_media_skRL = 1 - scoreRL.mean()
	error_std_skRL = scoreRL.std()

	print("Error medio LogisticRegression: " + str(error_media_skRL))
	print("Desviación media del error LogisticRegression: " + str(error_std_skRL))

	print("\nSGDClassifier")

	clfSGDC = SGDClassifier(max_iter=100, learning_rate='constant', eta0=1)
	scoreSGDC = cross_val_score(clfSGDC, X, Y, cv=10, n_jobs=-1)
	error_media_skSGDC = 1 - scoreSGDC.mean()
	error_std_skSGDC = scoreSGDC.std()

	print("Error medio SGDClassifier: " + str(error_media_skSGDC))
	print("Desviación media del error SGDClassifier: " + str(error_std_skSGDC))
	
	####################################################################################################################
	
	print("Wdbc:\n")
	print("nominalAtributos:")
	print(wdbc.nominalAtributos)
	print("\nDiccionario:")
	print(wdbc.diccionario)
	print("\nDatos:")
	print(wdbc.datos)

	####################################################################################################################

	print("************Knn SKLEARN************\n")
	print("\nK = 1\n")
	clf = KNeighborsClassifier(n_neighbors=1, p=2, metric='euclidean')
	score = cross_val_score(clf, X2, Y2, cv=10, n_jobs=-1)
	error_media_sk = 1 - score.mean()
	error_std_sk = score.std()
	print("Error medio sklearn euclidean: " + str(error_media_sk))
	print("Desviación media del error sklearn euclidean: " + str(error_std_sk))

	clfM = KNeighborsClassifier(n_neighbors=1, p=2, metric='manhattan')
	scoreM = cross_val_score(clfM, X2, Y2, cv=10, n_jobs=-1)
	error_media_skM = 1 - scoreM.mean()
	error_std_skM = scoreM.std()
	print("Error medio sklearn manhattan: " + str(error_media_skM))
	print("Desviación media del error sklearn manhattan: " + str(error_std_skM))

	clfMahalan = KNeighborsClassifier(n_neighbors=1, p=2, metric='mahalanobis', metric_params={'V': np.cov(x2)})
	scoreMahalan = cross_val_score(clfMahalan, X2, Y2, cv=10, n_jobs=-1)
	error_media_skMahalan = 1 - scoreMahalan.mean()
	error_std_skMahalan = scoreMahalan.std()
	print("Error medio sklearn mahalanobis: " + str(error_media_skMahalan))
	print("Desviación media del error sklearn mahalanobis: " + str(error_std_skMahalan))

	print("\nK = 3\n")
	clf3 = KNeighborsClassifier(n_neighbors=3, p=2, metric='euclidean')
	score3 = cross_val_score(clf3, X2, Y2, cv=10, n_jobs=-1)
	error_media_sk3 = 1 - score3.mean()
	error_std_sk3 = score3.std()
	print("Error medio sklearn euclidean: " + str(error_media_sk3))
	print("Desviación media del error sklearn euclidean: " + str(error_std_sk3))

	clf3M = KNeighborsClassifier(n_neighbors=3, p=2, metric='manhattan')
	score3M = cross_val_score(clf3M, X2, Y2, cv=10, n_jobs=-1)
	error_media_sk3M = 1 - score3M.mean()
	error_std_sk3M = score3M.std()
	print("Error medio sklearn manhattan: " + str(error_media_sk3M))
	print("Desviación media del error sklearn manhattan: " + str(error_std_sk3M))

	clf3Mahalan = KNeighborsClassifier(n_neighbors=3, p=2, metric='mahalanobis', metric_params={'V': np.cov(x2)})
	score3Mahalan = cross_val_score(clf3Mahalan, X2, Y2, cv=10, n_jobs=-1)
	error_media_sk3Mahalan = 1 - score3Mahalan.mean()
	error_std_sk3Mahalan = score3Mahalan.std()
	print("Error medio sklearn mahalanobis: " + str(error_media_sk3Mahalan))
	print("Desviación media del error sklearn mahalanobis: " + str(error_std_sk3Mahalan))


	print("\nK = 5\n")
	clf5 = KNeighborsClassifier(n_neighbors=5, p=2, metric='euclidean')
	score5 = cross_val_score(clf5, X2, Y2, cv=10, n_jobs=-1)
	error_media_sk5 = 1 - score5.mean()
	error_std_sk5 = score5.std()
	print("Error medio sklearn euclidean: " + str(error_media_sk5))
	print("Desviación media del error sklearn euclidean: " + str(error_std_sk5))

	clf5M = KNeighborsClassifier(n_neighbors=5, p=2, metric='manhattan')
	score5M = cross_val_score(clf5M, X2, Y2, cv=10, n_jobs=-1)
	error_media_sk5M = 1 - score5M.mean()
	error_std_sk5M = score5M.std()
	print("Error medio sklearn manhattan: " + str(error_media_sk5M))
	print("Desviación media del error sklearn manhattan: " + str(error_std_sk5M))

	clf5Mahalan = KNeighborsClassifier(n_neighbors=5, p=2, metric='mahalanobis', metric_params={'V': np.cov(x2)})
	score5Mahalan = cross_val_score(clf5Mahalan, X2, Y2, cv=10, n_jobs=-1)
	error_media_sk5Mahalan = 1 - score5Mahalan.mean()
	error_std_sk5Mahalan = score5Mahalan.std()
	print("Error medio sklearn mahalanobis: " + str(error_media_sk5Mahalan))
	print("Desviación media del error sklearn mahalanobis: " + str(error_std_sk5Mahalan))

	print("\nK = 11\n")
	clf11 = KNeighborsClassifier(n_neighbors=11, p=2, metric='euclidean')
	score11 = cross_val_score(clf11, X2, Y2, cv=10, n_jobs=-1)
	error_media_sk11 = 1 - score11.mean()
	error_std_sk11 = score11.std()
	print("Error medio sklearn euclidean: " + str(error_media_sk11))
	print("Desviación media del error sklearn euclidean: " + str(error_std_sk11))

	clf11M = KNeighborsClassifier(n_neighbors=11, p=2, metric='manhattan')
	score11M = cross_val_score(clf11M, X2, Y2, cv=10, n_jobs=-1)
	error_media_sk11M = 1 - score11M.mean()
	error_std_sk11M = score11M.std()
	print("Error medio sklearn manhattan: " + str(error_media_sk11M))
	print("Desviación media del error sklearn manhattan: " + str(error_std_sk11M))

	clf11Mahalan = KNeighborsClassifier(n_neighbors=11, p=2, metric='mahalanobis', metric_params={'V': np.cov(x2)})
	score11Mahalan = cross_val_score(clf11Mahalan, X2, Y2, cv=10, n_jobs=-1)
	error_media_sk11Mahalan = 1 - score11Mahalan.mean()
	error_std_sk11Mahalan = score11Mahalan.std()
	print("Error medio sklearn mahalanobis: " + str(error_media_sk11Mahalan))
	print("Desviación media del error sklearn mahalanobis: " + str(error_std_sk11Mahalan))

	print("\nK = 21\n")
	clf21 = KNeighborsClassifier(n_neighbors=21, p=2, metric='euclidean')
	score21 = cross_val_score(clf21, X2, Y2, cv=10, n_jobs=-1)
	error_media_sk21 = 1 - score21.mean()
	error_std_sk21 = score21.std()
	print("Error medio sklearn euclidean: " + str(error_media_sk21))
	print("Desviación media del error sklearn euclidean: " + str(error_std_sk21))

	clf21M = KNeighborsClassifier(n_neighbors=21, p=2, metric='manhattan')
	score21M = cross_val_score(clf21M, X2, Y2, cv=10, n_jobs=-1)
	error_media_sk21M = 1 - score21M.mean()
	error_std_sk21M = score21M.std()
	print("Error medio sklearn manhattan: " + str(error_media_sk21M))
	print("Desviación media del error sklearn manhattan: " + str(error_std_sk21M))

	clf21Mahalan = KNeighborsClassifier(n_neighbors=21, p=2, metric='mahalanobis', metric_params={'V': np.cov(x2)})
	score21Mahalan = cross_val_score(clf21Mahalan, X2, Y2, cv=10, n_jobs=-1)
	error_media_sk21Mahalan = 1 - score21Mahalan.mean()
	error_std_sk21Mahalan = score21Mahalan.std()
	print("Error medio sklearn mahalanobis: " + str(error_media_sk21Mahalan))
	print("Desviación media del error sklearn mahalanobis: " + str(error_std_sk21Mahalan))

########################################################################################################################

	print("\n*************** Regresión Logística ***************")
	print("\nLogisticRegression")
	clfRL = LogisticRegression(max_iter=10000, fit_intercept=1)
	scoreRL = cross_val_score(clfRL, X2, Y2, cv=10, n_jobs=-1)
	error_media_skRL = 1 - scoreRL.mean()
	error_std_skRL = scoreRL.std()

	print("Error medio LogisticRegression: " + str(error_media_skRL))
	print("Desviación media del error LogisticRegression: " + str(error_std_skRL))

	print("\nSGDClassifier")

	clfSGDC = SGDClassifier(max_iter=100, learning_rate='constant', eta0=1)
	scoreSGDC = cross_val_score(clfSGDC, X2, Y2, cv=10, n_jobs=-1)
	error_media_skSGDC = 1 - scoreSGDC.mean()
	error_std_skSGDC = scoreSGDC.std()

	print("Error medio SGDClassifier: " + str(error_media_skSGDC))
	print("Desviación media del error SGDClassifier: " + str(error_std_skSGDC))

########################################################################################################################
########################################################################################################################

	# Análisis ROC
	diabetes = Datos("pima-indians-diabetes.data")
	wdbc = Datos("wdbc.data")
	euclid_1_x = [0, 0, 1]
	euclid_1_y = [0, 0, 1]
	euclid_5_x = [0, 0, 1]
	euclid_5_y = [0, 0, 1]
	euclid_11_x = [0, 0, 1]
	euclid_11_y = [0, 0, 1]
	euclid_21_x = [0, 0, 1]
	euclid_21_y = [0, 0, 1]
	manhatt_1_x = [0, 0, 1]
	manhatt_1_y = [0, 0, 1]
	manhatt_5_x = [0, 0, 1]
	manhatt_5_y = [0, 0, 1]
	manhatt_11_x = [0, 0, 1]
	manhatt_11_y = [0, 0, 1]
	manhatt_21_x = [0, 0, 1]
	manhatt_21_y = [0, 0, 1]
	mahalan_1_x = [0, 0, 1]
	mahalan_1_y = [0, 0, 1]
	mahalan_5_x = [0, 0, 1]
	mahalan_5_y = [0, 0, 1]
	mahalan_11_x = [0, 0, 1]
	mahalan_11_y = [0, 0, 1]
	mahalan_21_x = [0, 0, 1]
	mahalan_21_y = [0, 0, 1]

	# K-NN -> Diabetes
	knn = ClasificadorVecinosProximos()
	vc = ValidacionCruzada()
	vc.creaParticiones(diabetes.datos)
	particiones = vc.particiones
	for particion in particiones:
		knn.entrenamiento(diabetes.extraeDatos(particion.indicesTrain), diabetes.nominalAtributos, diabetes.diccionario)
		datosTest = diabetes.extraeDatos(particion.indicesTest)
		result = knn.clasifica(datosTest, diabetes.nominalAtributos, diabetes.diccionario)
		clase = datosTest[:, len(diabetes.nominalAtributos) - 1]
		for i in range(0, len(result)):
			tp, tn, fp, fn = valores_roc(clase, result[i])

			# i va de 0 a 1
			# i = 0, 1, 2 => k = 1
			# i = 3, 4, 5 => k = 5
			# i = 6, 7, 8 => k = 11
			# i = 9, 10, 11 => k = 21
			# i = 0, 3, 6, 9 => Distancia Euclidea
			# i = 1, 4, 7, 10 => Distancia de Manhattan
			# i = 2, 5, 8, 11 => Distancia de Mahalanobis
			tpr = tp / (tp + fn)
			fpr = fp / (fp + tn)

			if i == 0:
				euclid_1_x[1] += fpr
				euclid_1_y[1] += tpr
			elif i == 1:
				manhatt_1_x[1] += fpr
				manhatt_1_y[1] += tpr
			elif i == 2:
				mahalan_1_x[1] += fpr
				mahalan_1_y[1] += tpr
			elif i == 3:
				euclid_5_x[1] += fpr
				euclid_5_y[1] += tpr
			elif i == 4:
				manhatt_5_x[1] += fpr
				manhatt_5_y[1] += tpr
			elif i == 5:
				mahalan_5_x[1] += fpr
				mahalan_5_y[1] += tpr
			elif i == 6:
				euclid_11_x[1] += fpr
				euclid_11_y[1] += tpr
			elif i == 7:
				manhatt_11_x[1] += fpr
				manhatt_11_y[1] += tpr
			elif i == 8:
				mahalan_11_x[1] += fpr
				mahalan_11_y[1] += tpr
			elif i == 9:
				euclid_21_x[1] += fpr
				euclid_21_y[1] += tpr
			elif i == 10:
				manhatt_21_x[1] += fpr
				manhatt_21_y[1] += tpr
			elif i == 11:
				mahalan_21_x[1] += fpr
				mahalan_21_y[1] += tpr

	euclid_1_x[1] /= len(particiones)
	euclid_1_y[1] /= len(particiones)
	euclid_5_x[1] /= len(particiones)
	euclid_5_y[1] /= len(particiones)
	euclid_11_x[1] /= len(particiones)
	euclid_11_y[1] /= len(particiones)
	euclid_21_x[1] /= len(particiones)
	euclid_21_y[1] /= len(particiones)
	manhatt_1_x[1] /= len(particiones)
	manhatt_1_y[1] /= len(particiones)
	manhatt_5_x[1] /= len(particiones)
	manhatt_5_y[1] /= len(particiones)
	manhatt_11_x[1] /= len(particiones)
	manhatt_11_y[1] /= len(particiones)
	manhatt_21_x[1] /= len(particiones)
	manhatt_21_y[1] /= len(particiones)
	mahalan_1_x[1] /= len(particiones)
	mahalan_1_y[1] /= len(particiones)
	mahalan_5_x[1] /= len(particiones)
	mahalan_5_y[1] /= len(particiones)
	mahalan_11_x[1] /= len(particiones)
	mahalan_11_y[1] /= len(particiones)
	mahalan_21_x[1] /= len(particiones)
	mahalan_21_y[1] /= len(particiones)

	repr_grafica('Curva ROC Diabetes K-NN k=1', 'False Positive Rate', 'True Positive Rate', 3,
				 ['euclid', 'manhattan', 'mahalanobis'], ['darkorange', 'darkgreen', 'deeppink'],
				 [euclid_1_x, manhatt_1_x, mahalan_1_x], [euclid_1_y, manhatt_1_y, mahalan_1_y])
	repr_grafica('Curva ROC Diabetes K-NN k=5', 'False Positive Rate', 'True Positive Rate', 3,
				 ['euclid', 'manhattan', 'mahalanobis'], ['darkorange', 'darkgreen', 'deeppink'],
				 [euclid_5_x, manhatt_5_x, mahalan_5_x], [euclid_5_y, manhatt_5_y, mahalan_5_y])
	repr_grafica('Curva ROC  Diabetes K-NN k=11', 'False Positive Rate', 'True Positive Rate', 3,
				 ['euclid', 'manhattan', 'mahalanobis'], ['darkorange', 'darkgreen', 'deeppink'],
				 [euclid_11_x, manhatt_11_x, mahalan_11_x], [euclid_11_y, manhatt_11_y, mahalan_11_y])
	repr_grafica('Curva ROC Diabetes K-NN k=21', 'False Positive Rate', 'True Positive Rate', 3,
				 ['euclid', 'manhattan', 'mahalanobis'], ['darkorange', 'darkgreen', 'deeppink'],
				 [euclid_21_x, manhatt_21_x, mahalan_21_x], [euclid_21_y, manhatt_21_y, mahalan_21_y])

	euclid_1_x = [0, 0, 1]
	euclid_1_y = [0, 0, 1]
	euclid_5_x = [0, 0, 1]
	euclid_5_y = [0, 0, 1]
	euclid_11_x = [0, 0, 1]
	euclid_11_y = [0, 0, 1]
	euclid_21_x = [0, 0, 1]
	euclid_21_y = [0, 0, 1]
	manhatt_1_x = [0, 0, 1]
	manhatt_1_y = [0, 0, 1]
	manhatt_5_x = [0, 0, 1]
	manhatt_5_y = [0, 0, 1]
	manhatt_11_x = [0, 0, 1]
	manhatt_11_y = [0, 0, 1]
	manhatt_21_x = [0, 0, 1]
	manhatt_21_y = [0, 0, 1]
	mahalan_1_x = [0, 0, 1]
	mahalan_1_y = [0, 0, 1]
	mahalan_5_x = [0, 0, 1]
	mahalan_5_y = [0, 0, 1]
	mahalan_11_x = [0, 0, 1]
	mahalan_11_y = [0, 0, 1]
	mahalan_21_x = [0, 0, 1]
	mahalan_21_y = [0, 0, 1]

	# K-NN -> WDBC
	knn = ClasificadorVecinosProximos()
	vc = ValidacionCruzada()
	vc.creaParticiones(wdbc.datos)
	particiones = vc.particiones
	for particion in particiones:
		knn.entrenamiento(wdbc.extraeDatos(particion.indicesTrain), wdbc.nominalAtributos, wdbc.diccionario)
		datosTest = wdbc.extraeDatos(particion.indicesTest)
		result = knn.clasifica(datosTest, wdbc.nominalAtributos, wdbc.diccionario)
		clase = datosTest[:, len(wdbc.nominalAtributos) - 1]
		for i in range(0, len(result)):
			tp, tn, fp, fn = valores_roc(clase, result[i])

			# i va de 0 a 1
			# i = 0, 1, 2 => k = 1
			# i = 3, 4, 5 => k = 5
			# i = 6, 7, 8 => k = 11
			# i = 9, 10, 11 => k = 21
			# i = 0, 3, 6, 9 => Distancia Euclidea
			# i = 1, 4, 7, 10 => Distancia de Manhattan
			# i = 2, 5, 8, 11 => Distancia de Mahalanobis
			tpr = tp / (tp + fn)
			fpr = fp / (fp + tn)

			if i == 0:
				euclid_1_x[1] += fpr
				euclid_1_y[1] += tpr
			elif i == 1:
				manhatt_1_x[1] += fpr
				manhatt_1_y[1] += tpr
			elif i == 2:
				mahalan_1_x[1] += fpr
				mahalan_1_y[1] += tpr
			elif i == 3:
				euclid_5_x[1] += fpr
				euclid_5_y[1] += tpr
			elif i == 4:
				manhatt_5_x[1] += fpr
				manhatt_5_y[1] += tpr
			elif i == 5:
				mahalan_5_x[1] += fpr
				mahalan_5_y[1] += tpr
			elif i == 6:
				euclid_11_x[1] += fpr
				euclid_11_y[1] += tpr
			elif i == 7:
				manhatt_11_x[1] += fpr
				manhatt_11_y[1] += tpr
			elif i == 8:
				mahalan_11_x[1] += fpr
				mahalan_11_y[1] += tpr
			elif i == 9:
				euclid_21_x[1] += fpr
				euclid_21_y[1] += tpr
			elif i == 10:
				manhatt_21_x[1] += fpr
				manhatt_21_y[1] += tpr
			elif i == 11:
				mahalan_21_x[1] += fpr
				mahalan_21_y[1] += tpr

	euclid_1_x[1] /= len(particiones)
	euclid_1_y[1] /= len(particiones)
	euclid_5_x[1] /= len(particiones)
	euclid_5_y[1] /= len(particiones)
	euclid_11_x[1] /= len(particiones)
	euclid_11_y[1] /= len(particiones)
	euclid_21_x[1] /= len(particiones)
	euclid_21_y[1] /= len(particiones)
	manhatt_1_x[1] /= len(particiones)
	manhatt_1_y[1] /= len(particiones)
	manhatt_5_x[1] /= len(particiones)
	manhatt_5_y[1] /= len(particiones)
	manhatt_11_x[1] /= len(particiones)
	manhatt_11_y[1] /= len(particiones)
	manhatt_21_x[1] /= len(particiones)
	manhatt_21_y[1] /= len(particiones)
	mahalan_1_x[1] /= len(particiones)
	mahalan_1_y[1] /= len(particiones)
	mahalan_5_x[1] /= len(particiones)
	mahalan_5_y[1] /= len(particiones)
	mahalan_11_x[1] /= len(particiones)
	mahalan_11_y[1] /= len(particiones)
	mahalan_21_x[1] /= len(particiones)
	mahalan_21_y[1] /= len(particiones)

	repr_grafica('Curva ROC WDBC K-NN k=1', 'False Positive Rate', 'True Positive Rate', 3,
				 ['euclid', 'manhattan', 'mahalanobis'], ['darkorange', 'darkgreen', 'deeppink'],
				 [euclid_1_x, manhatt_1_x, mahalan_1_x], [euclid_1_y, manhatt_1_y, mahalan_1_y])
	repr_grafica('Curva ROC WDBC K-NN k=5', 'False Positive Rate', 'True Positive Rate', 3,
				 ['euclid', 'manhattan', 'mahalanobis'], ['darkorange', 'darkgreen', 'deeppink'],
				 [euclid_5_x, manhatt_5_x, mahalan_5_x], [euclid_5_y, manhatt_5_y, mahalan_5_y])
	repr_grafica('Curva ROC  WDBC K-NN k=11', 'False Positive Rate', 'True Positive Rate', 3,
				 ['euclid', 'manhattan', 'mahalanobis'], ['darkorange', 'darkgreen', 'deeppink'],
				 [euclid_11_x, manhatt_11_x, mahalan_11_x], [euclid_11_y, manhatt_11_y, mahalan_11_y])
	repr_grafica('Curva ROC WDBC K-NN k=21', 'False Positive Rate', 'True Positive Rate', 3,
				 ['euclid', 'manhattan', 'mahalanobis'], ['darkorange', 'darkgreen', 'deeppink'],
				 [euclid_21_x, manhatt_21_x, mahalan_21_x], [euclid_21_y, manhatt_21_y, mahalan_21_y])

	####################################################################################################################
	x = [0, 0, 1]
	y = [0, 0, 1]
	# Regresión -> Diabetes
	reg = ClasificadorRegresionLogistica()
	vc = ValidacionCruzada()
	vc.creaParticiones(diabetes.datos)
	particiones = vc.particiones
	for particion in particiones:
		reg.entrenamiento(diabetes.extraeDatos(particion.indicesTrain), diabetes.nominalAtributos, diabetes.diccionario)
		datosTest = diabetes.extraeDatos(particion.indicesTest)
		result = reg.clasifica(datosTest, diabetes.nominalAtributos, diabetes.diccionario)
		clase = datosTest[:, len(diabetes.nominalAtributos) - 1]
		for i in range(0, len(result)):
			tp, tn, fp, fn = valores_roc(clase, result[i])
			tpr = tp / (tp + fn)
			fpr = fp / (fp + tn)
			x[1] += fpr
			y[1] += tpr
	x[1] /= len(particiones)
	y[1] /= len(particiones)
	repr_grafica('Curva ROC Regresión Diabetes', 'False Positive Rate', 'True Positive Rate', 1,['Log_reg'],
				 ['darkorange'], [x], [y])

	x = [0, 0, 1]
	y = [0, 0, 1]
	# Regresión -> WDBC
	reg = ClasificadorRegresionLogistica()
	vc = ValidacionCruzada()
	vc.creaParticiones(wdbc.datos)
	particiones = vc.particiones
	for particion in particiones:
		reg.entrenamiento(wdbc.extraeDatos(particion.indicesTrain), wdbc.nominalAtributos, wdbc.diccionario)
		datosTest = wdbc.extraeDatos(particion.indicesTest)
		result = reg.clasifica(datosTest, wdbc.nominalAtributos, wdbc.diccionario)
		clase = datosTest[:, len(wdbc.nominalAtributos) - 1]
		for i in range(0, len(result)):
			tp, tn, fp, fn = valores_roc(clase, result[i])
			tpr = tp / (tp + fn)
			fpr = fp / (fp + tn)
			x[1] += fpr
			y[1] += tpr
	x[1] /= len(particiones)
	y[1] /= len(particiones)
	repr_grafica('Curva ROC Regresión WDBC', 'False Positive Rate', 'True Positive Rate', 1, ['Log_reg'],
				 ['darkorange'], [x], [y])


# Representar puntos

if __name__ == '__main__':
	main()
