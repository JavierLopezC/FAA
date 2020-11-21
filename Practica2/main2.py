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

from EstrategiaParticionado import EstrategiaParticionado


def error_sk(data, pred):
	tot = 0
	for i in range(0, len(pred)):
		if pred[i] != data[i]:
			tot += 1
	tot /= len(pred)
	# print("error de: " + str(tot))
	return tot


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

	encAtributos = ColumnTransformer([("diabetes", OneHotEncoder(), [0])], remainder="passthrough")
	X=encAtributos.fit_transform(diabetes.datos[:, :-1])
	Y = diabetes.datos[:, -1]

	encAtributos2 = ColumnTransformer([("wdbc", OneHotEncoder(), [0])], remainder="passthrough")
	X2 = encAtributos2.fit_transform(wdbc.datos[:, :-1])
	Y2 = wdbc.datos[:, -1]

	#estrategia = EstrategiaParticionado.ValidacionCruzada()
	
	print("\n\nDiabetes:")
	
	#print("\nValidacion Cruzada NB")
	#print("\nValidando con clasificador propio:")
	#nb = ClasificadorNaiveBayes()
	#vc = ValidacionCruzada()
	#error = Clasificador.validacion(vc, diabetes, nb)
	#print("Error medio: " + str(error))
	
	#######################################################################################################################
	print("************Knn SKLEARN************\n")
	print("\nK = 1\n")
	clf = KNeighborsClassifier(n_neighbors=1, p=2, metric='euclidean')
	score = cross_val_score(clf, X, Y, cv=10, n_jobs=-1)
	error_media_sk = 1 - score.mean()
	error_std_sk = score.std()
	print("Error medio sklearn euclidean: " + str(error_media_sk))
	print("Score medio sklearn euclidean: " + str(error_std_sk))

	clfM = KNeighborsClassifier(n_neighbors=1, p=2, metric='manhattan')
	scoreM = cross_val_score(clfM, X, Y, cv=10, n_jobs=-1)
	error_media_skM = 1 - scoreM.mean()
	error_std_skM = scoreM.std()
	print("Error medio sklearn manhattan: " + str(error_media_skM))
	print("Score medio sklearn manhattan: " + str(error_std_skM))

	print("\nK = 3\n")
	clf3 = KNeighborsClassifier(n_neighbors=3, p=2, metric='euclidean')
	score3 = cross_val_score(clf3, X, Y, cv=10, n_jobs=-1)
	error_media_sk3 = 1 - score3.mean()
	error_std_sk3 = score3.std()
	print("Error medio sklearn euclidean: " + str(error_media_sk3))
	print("Score medio sklearn euclidean: " + str(error_std_sk3))

	clf3M = KNeighborsClassifier(n_neighbors=3, p=2, metric='manhattan')
	score3M = cross_val_score(clf3M, X, Y, cv=10, n_jobs=-1)
	error_media_sk3M = 1 - score3M.mean()
	error_std_sk3M = score3M.std()
	print("Error medio sklearn manhattan: " + str(error_media_sk3M))
	print("Score medio sklearn manhattan: " + str(error_std_sk3M))
	
	clf3Mahalan = KNeighborsClassifier(n_neighbors=3, p=2, metric='mahalanobis', metric_params={'V': np.cov(X, Y)})
	score3Mahalan = cross_val_score(clf3Mahalan, X, Y, cv=10, n_jobs=-1)
	error_media_sk3Mahalan = 1 - score3Mahalan.mean()
	error_std_sk3Mahalan = score3Mahalan.std()
	print("Error medio sklearn mahalanobis: " + str(error_media_sk3Mahalan))
	print("Score medio sklearn mahalanobis: " + str(error_std_sk3Mahalan))


	print("\nK = 5\n")
	clf5 = KNeighborsClassifier(n_neighbors=5, p=2, metric='euclidean')
	score5 = cross_val_score(clf5, X, Y, cv=10, n_jobs=-1)
	error_media_sk5 = 1 - score5.mean()
	error_std_sk5 = score5.std()
	print("Error medio sklearn euclidean: " + str(error_media_sk5))
	print("Score medio sklearn euclidean: " + str(error_std_sk5))

	clf5M = KNeighborsClassifier(n_neighbors=5, p=2, metric='manhattan')
	score5M = cross_val_score(clf5M, X, Y, cv=10, n_jobs=-1)
	error_media_sk5M = 1 - score5M.mean()
	error_std_sk5M = score5M.std()
	print("Error medio sklearn manhattan: " + str(error_media_sk5M))
	print("Score medio sklearn manhattan: " + str(error_std_sk5M))

	print("\nK = 11\n")
	clf11 = KNeighborsClassifier(n_neighbors=11, p=2, metric='euclidean')
	score11 = cross_val_score(clf11, X, Y, cv=10, n_jobs=-1)
	error_media_sk11 = 1 - score11.mean()
	error_std_sk11 = score11.std()
	print("Error medio sklearn euclidean: " + str(error_media_sk11))
	print("Score medio sklearn euclidean: " + str(error_std_sk11))

	clf11M = KNeighborsClassifier(n_neighbors=11, p=2, metric='manhattan')
	score11M = cross_val_score(clf11M, X, Y, cv=10, n_jobs=-1)
	error_media_sk11M = 1 - score11M.mean()
	error_std_sk11M = score11M.std()
	print("Error medio sklearn manhattan: " + str(error_media_sk11M))
	print("Score medio sklearn manhattan: " + str(error_std_sk11M))

	print("\nK = 21\n")
	clf21 = KNeighborsClassifier(n_neighbors=21, p=2, metric='euclidean')
	score21 = cross_val_score(clf21, X, Y, cv=10, n_jobs=-1)
	error_media_sk21 = 1 - score21.mean()
	error_std_sk21 = score21.std()
	print("Error medio sklearn euclidean: " + str(error_media_sk21))
	print("Score medio sklearn euclidean: " + str(error_std_sk21))

	clf21M = KNeighborsClassifier(n_neighbors=21, p=2, metric='manhattan')
	score21M = cross_val_score(clf21M, X, Y, cv=10, n_jobs=-1)
	error_media_sk21M = 1 - score21M.mean()
	error_std_sk21M = score21M.std()
	print("Error medio sklearn manhattan: " + str(error_media_sk21M))
	print("Score medio sklearn manhattan: " + str(error_std_sk21M))


#	print("\nValidación Simple K-NN")
#	print("\nValidando 100 veces con clasificador propio:")
#	knn = ClasificadorVecinosProximos()
#	vs = ValidacionSimple()
#	error = Clasificador.validacion(vs, diabetes, knn)
#	print("Error medio: " + str(error))

#	print("\nValidacion Cruzada K-NN")
#	print("\nValidando con clasificador propio:")
#	knn = ClasificadorVecinosProximos()
#	vc = ValidacionCruzada()
#	error = Clasificador.validacion(vc, diabetes, knn)
#	print("Error medio: " + str(error))
	
	#######################################################################################################################

	print("\n*************** Regresión Logística ***************")
	print("\nLogisticRegression")
	clfRL = LogisticRegression(max_iter=10000, fit_intercept=1)
	scoreRL = cross_val_score(clfRL, X, Y, cv=10, n_jobs=-1)
	error_media_skRL = 1 - scoreRL.mean()
	error_std_skRL = scoreRL.std()

	print("Error medio LogisticRegression: " + str(error_media_skRL))
	print("Score medio LogisticRegression: " + str(error_std_skRL))

	print("\nSGDClassifier")

	clfSGDC = SGDClassifier(max_iter=100, learning_rate='constant', eta0=1)
	scoreSGDC = cross_val_score(clfSGDC, X, Y, cv=10, n_jobs=-1)
	error_media_skSGDC = 1 - scoreSGDC.mean()
	error_std_skSGDC = scoreSGDC.std()

	print("Error medio SGDClassifier: " + str(error_media_skSGDC))
	print("Score medio SGDClassifier: " + str(error_std_skSGDC))

#	print("\nValidación Simple Regresión Logística")
#	print("\nValidando 100 veces con clasificador propio:")
#	reg = ClasificadorRegresionLogistica()
#	vs = ValidacionSimple()
#	error = Clasificador.validacion(vs, diabetes, reg)
#	print("Error medio: " + str(error))
	
#	print("\nValidacion Cruzada Regresión Logística")
#	print("\nValidando con clasificador propio:")
#	reg = ClasificadorRegresionLogistica()
#	vc = ValidacionCruzada()
#	error = Clasificador.validacion(vc, diabetes, reg)
#	print("Error medio: " + str(error))
	
	#######################################################################################################################
	
	print("Wdbc:\n")
	print("nominalAtributos:")
	print(wdbc.nominalAtributos)
	print("\nDiccionario:")
	print(wdbc.diccionario)
	print("\nDatos:")
	print(wdbc.datos)
	
#	print("\n\nWdbc:")
	# print("\nValidación Simple NB")
	
	# print("\nValidando 100 veces con clasificador propio:")
	# nb = ClasificadorNaiveBayes()
	# vs = ValidacionSimple()
	# error = Clasificador.validacion(vs, wdbc, nb)
	# print("Error medio: " + str(error))
	
	# print("\nValidacion Cruzada NB")
	# print("\nValidando con clasificador propio:")
	# nb = ClasificadorNaiveBayes()
	# vc = ValidacionCruzada()
	# error = Clasificador.validacion(vc, wdbc, nb)
	# print("Error medio: " + str(error))
	
	#######################################################################################################################
	
#	print("\nValidación Simple K-NN")
#	print("\nValidando 100 veces con clasificador propio:")
#	knn = ClasificadorVecinosProximos()
#	vs = ValidacionSimple()
#	error = Clasificador.validacion(vs, wdbc, knn)
#	print("Error medio: " + str(error))
#
#	print("\nValidacion Cruzada K-NN")
#	print("\nValidando con clasificador propio:")
#	knn = ClasificadorVecinosProximos()
#	vc = ValidacionCruzada()
#	error = Clasificador.validacion(vc, wdbc, knn)
#	print("Error medio: " + str(error))

	print("************Knn SKLEARN************\n")
	print("\nK = 1\n")
	clf = KNeighborsClassifier(n_neighbors=1, p=2, metric='euclidean')
	score = cross_val_score(clf, X2, Y2, cv=10, n_jobs=-1)
	error_media_sk = 1 - score.mean()
	error_std_sk = score.std()
	print("Error medio sklearn euclidean: " + str(error_media_sk))
	print("Score medio sklearn euclidean: " + str(error_std_sk))

	clfM = KNeighborsClassifier(n_neighbors=1, p=2, metric='manhattan')
	scoreM = cross_val_score(clfM, X2, Y2, cv=10, n_jobs=-1)
	error_media_skM = 1 - scoreM.mean()
	error_std_skM = scoreM.std()
	print("Error medio sklearn manhattan: " + str(error_media_skM))
	print("Score medio sklearn manhattan: " + str(error_std_skM))

	print("\nK = 3\n")
	clf3 = KNeighborsClassifier(n_neighbors=3, p=2, metric='euclidean')
	score3 = cross_val_score(clf3, X2, Y2, cv=10, n_jobs=-1)
	error_media_sk3 = 1 - score3.mean()
	error_std_sk3 = score3.std()
	print("Error medio sklearn euclidean: " + str(error_media_sk3))
	print("Score medio sklearn euclidean: " + str(error_std_sk3))

	clf3M = KNeighborsClassifier(n_neighbors=3, p=2, metric='manhattan')
	score3M = cross_val_score(clf3M, X2, Y2, cv=10, n_jobs=-1)
	error_media_sk3M = 1 - score3M.mean()
	error_std_sk3M = score3M.std()
	print("Error medio sklearn manhattan: " + str(error_media_sk3M))
	print("Score medio sklearn manhattan: " + str(error_std_sk3M))


	print("\nK = 5\n")
	clf5 = KNeighborsClassifier(n_neighbors=5, p=2, metric='euclidean')
	score5 = cross_val_score(clf5, X2, Y2, cv=10, n_jobs=-1)
	error_media_sk5 = 1 - score5.mean()
	error_std_sk5 = score5.std()
	print("Error medio sklearn euclidean: " + str(error_media_sk5))
	print("Score medio sklearn euclidean: " + str(error_std_sk5))

	clf5M = KNeighborsClassifier(n_neighbors=5, p=2, metric='manhattan')
	score5M = cross_val_score(clf5M, X2, Y2, cv=10, n_jobs=-1)
	error_media_sk5M = 1 - score5M.mean()
	error_std_sk5M = score5M.std()
	print("Error medio sklearn manhattan: " + str(error_media_sk5M))
	print("Score medio sklearn manhattan: " + str(error_std_sk5M))

	print("\nK = 11\n")
	clf11 = KNeighborsClassifier(n_neighbors=11, p=2, metric='euclidean')
	score11 = cross_val_score(clf11, X2, Y2, cv=10, n_jobs=-1)
	error_media_sk11 = 1 - score11.mean()
	error_std_sk11 = score11.std()
	print("Error medio sklearn euclidean: " + str(error_media_sk11))
	print("Score medio sklearn euclidean: " + str(error_std_sk11))

	clf11M = KNeighborsClassifier(n_neighbors=11, p=2, metric='manhattan')
	score11M = cross_val_score(clf11M, X2, Y2, cv=10, n_jobs=-1)
	error_media_sk11M = 1 - score11M.mean()
	error_std_sk11M = score11M.std()
	print("Error medio sklearn manhattan: " + str(error_media_sk11M))
	print("Score medio sklearn manhattan: " + str(error_std_sk11M))

	print("\nK = 21\n")
	clf21 = KNeighborsClassifier(n_neighbors=21, p=2, metric='euclidean')
	score21 = cross_val_score(clf21, X2, Y2, cv=10, n_jobs=-1)
	error_media_sk21 = 1 - score21.mean()
	error_std_sk21 = score21.std()
	print("Error medio sklearn euclidean: " + str(error_media_sk21))
	print("Score medio sklearn euclidean: " + str(error_std_sk21))

	clf21M = KNeighborsClassifier(n_neighbors=21, p=2, metric='manhattan')
	score21M = cross_val_score(clf21M, X2, Y2, cv=10, n_jobs=-1)
	error_media_sk21M = 1 - score21M.mean()
	error_std_sk21M = score21M.std()
	print("Error medio sklearn manhattan: " + str(error_media_sk21M))
	print("Score medio sklearn manhattan: " + str(error_std_sk21M))

#######################################################################################################################

# print("\nValidación Simple Regresión Logística")
# print("\nValidando 100 veces con clasificador propio:")
# reg = ClasificadorRegresionLogistica()
# vs = ValidacionSimple()
# error = Clasificador.validacion(vs, wdbc, reg)
# print("Error medio: " + str(error))

# print("\nValidacion Cruzada Regresión Logística")
# print("\nValidando con clasificador propio:")
# reg = ClasificadorRegresionLogistica()
# vc = ValidacionCruzada()
# error = Clasificador.validacion(vc, wdbc, reg)
# print("Error medio: " + str(error))

	print("\n*************** Regresión Logística ***************")
	print("\nLogisticRegression")
	clfRL = LogisticRegression(max_iter=10000, fit_intercept=1)
	scoreRL = cross_val_score(clfRL, X2, Y2, cv=10, n_jobs=-1)
	error_media_skRL = 1 - scoreRL.mean()
	error_std_skRL = scoreRL.std()

	print("Error medio LogisticRegression: " + str(error_media_skRL))
	print("Score medio LogisticRegression: " + str(error_std_skRL))

	print("\nSGDClassifier")

	clfSGDC = SGDClassifier(max_iter=100, learning_rate='constant', eta0=1)
	scoreSGDC = cross_val_score(clfSGDC, X2, Y2, cv=10, n_jobs=-1)
	error_media_skSGDC = 1 - scoreSGDC.mean()
	error_std_skSGDC = scoreSGDC.std()

	print("Error medio SGDClassifier: " + str(error_media_skSGDC))
	print("Score medio SGDClassifier: " + str(error_std_skSGDC))
#######################################################################################################################

if __name__ == '__main__':
	main()
