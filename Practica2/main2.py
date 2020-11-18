# -*- coding: utf-8 -*-

# coding: utf-8
from Datos import Datos
import numpy as np
from Clasificador import Clasificador, ClasificadorNaiveBayes, ClasificadorVecinosProximos, ClasificadorRegresionLogistica
from EstrategiaParticionado import ValidacionSimple, ValidacionCruzada
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.naive_bayes import GaussianNB, MultinomialNB


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
	
	print("\n\nDiabetes:")
	#print("\nValidación Simple NB")
	
	#print("\nValidando 100 veces con clasificador propio:")
	#nb = ClasificadorNaiveBayes()
	#vs = ValidacionSimple()
	#error = Clasificador.validacion(vs, diabetes, nb)
	#print("Error medio: " + str(error))
	
	#print("\nValidacion Cruzada NB")
	#print("\nValidando con clasificador propio:")
	#nb = ClasificadorNaiveBayes()
	#vc = ValidacionCruzada()
	#error = Clasificador.validacion(vc, diabetes, nb)
	#print("Error medio: " + str(error))
	
	#######################################################################################################################
	
	print("\nValidación Simple K-NN")
	print("\nValidando 100 veces con clasificador propio:")
	knn = ClasificadorVecinosProximos()
	vs = ValidacionSimple()
	error = Clasificador.validacion(vs, diabetes, knn)
	print("Error medio: " + str(error))
	
	print("\nValidacion Cruzada K-NN")
	print("\nValidando con clasificador propio:")
	knn = ClasificadorVecinosProximos()
	vc = ValidacionCruzada()
	error = Clasificador.validacion(vc, diabetes, knn)
	print("Error medio: " + str(error))
	
	#######################################################################################################################
	
	#print("\nValidación Simple Regresión Logística")
	#print("\nValidando 100 veces con clasificador propio:")
	#reg = ClasificadorRegresionLogistica()
	#vs = ValidacionSimple()
	#error = Clasificador.validacion(vs, diabetes, reg)
	#print("Error medio: " + str(error))
	
	#print("\nValidacion Cruzada Regresión Logística")
	#print("\nValidando con clasificador propio:")
	#reg = ClasificadorRegresionLogistica()
	#vc = ValidacionCruzada()
	#error = Clasificador.validacion(vc, diabetes, reg)
	#print("Error medio: " + str(error))
	
	#######################################################################################################################
	
	print("Wdbc:\n")
	print("nominalAtributos:")
	print(wdbc.nominalAtributos)
	print("\nDiccionario:")
	print(wdbc.diccionario)
	print("\nDatos:")
	print(wdbc.datos)
	
	print("\n\nWdbc:")
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
	
	print("\nValidación Simple K-NN")
	print("\nValidando 100 veces con clasificador propio:")
	knn = ClasificadorVecinosProximos()
	vs = ValidacionSimple()
	error = Clasificador.validacion(vs, wdbc, knn)
	print("Error medio: " + str(error))
	
	print("\nValidacion Cruzada K-NN")
	print("\nValidando con clasificador propio:")
	knn = ClasificadorVecinosProximos()
	vc = ValidacionCruzada()
	error = Clasificador.validacion(vc, wdbc, knn)
	print("Error medio: " + str(error))


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

#######################################################################################################################

if __name__ == '__main__':
	main()
