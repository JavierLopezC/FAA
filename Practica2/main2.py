# -*- coding: utf-8 -*-

# coding: utf-8
from Datos import Datos
import numpy as np
from Clasificador import Clasificador
from Clasificador import ClasificadorNaiveBayes, ClasificadorVecinosProximos, ClasificadorRegresionLogistica
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
	
	print("\n\nDiabetes:")
	
	print("\nValidacion Cruzada NB")
	print("\nValidando con clasificador propio:")
	nb = ClasificadorNaiveBayes()
	vc = ValidacionCruzada()
	error = Clasificador.validacion(vc, diabetes, nb)
	print("Error medio: " + str(error))
	
	#######################################################################################################################
	
	print("\nValidacion Cruzada K-NN")
	print("\nValidando con clasificador propio:")
	knn = ClasificadorVecinosProximos()
	vc = ValidacionCruzada()
	error = Clasificador.validacion(vc, diabetes, knn)
	print("Error medio: " + str(error))
	
	#######################################################################################################################
	
	print("\nValidacion Cruzada Regresión Logística")
	print("\nValidando con clasificador propio:")
	reg = ClasificadorRegresionLogistica()
	vc = ValidacionCruzada()
	error = Clasificador.validacion(vc, diabetes, reg)
	print("Error medio: " + str(error))
	
	#######################################################################################################################
	
	print("Wdbc:\n")
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
	print("Error medio: " + str(error))
	
	#######################################################################################################################
	
	print("\nValidacion Cruzada K-NN")
	print("\nValidando con clasificador propio:")
	knn = ClasificadorVecinosProximos()
	vc = ValidacionCruzada()
	error = Clasificador.validacion(vc, wdbc, knn)
	print("Error medio: " + str(error))

#######################################################################################################################

	print("\nValidacion Cruzada Regresión Logística")
	print("\nValidando con clasificador propio:")
	reg = ClasificadorRegresionLogistica()
	vc = ValidacionCruzada()
	error = Clasificador.validacion(vc, wdbc, reg)
	print("Error medio: " + str(error))

#######################################################################################################################
#######################################################################################################################

	# Análisis ROC
	diabetes = Datos("pima-indians-diabetes.data")
	wdbc = Datos("wdbc.data")
	
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
				euclid_1_tpr = tpr
				euclid_1_fpr = fpr
			elif i == 1:
				manhatt_1_tpr = tpr
				manhatt_1_fpr = fpr
			elif i == 2:
				mahalan_1_tpr = tpr
				mahalan_1_fpr = fpr
			elif i == 3:
				euclid_5_tpr = tpr
				euclid_5_fpr = fpr
			elif i == 4:
				manhatt_5_tpr = tpr
				manhatt_5_fpr = fpr
			elif i == 5:
				mahalan_5_tpr = tpr
				mahalan_5_fpr = fpr
			elif i == 6:
				euclid_11_tpr = tpr
				euclid_11_fpr = fpr
			elif i == 7:
				manhatt_11_tpr = tpr
				manhatt_11_fpr = fpr
			elif i == 8:
				mahalan_11_tpr = tpr
				mahalan_11_fpr = fpr
			elif i == 9:
				euclid_21_tpr = tpr
				euclid_21_fpr = fpr
			elif i == 10:
				manhatt_21_tpr = tpr
				manhatt_21_fpr = fpr
			elif i == 11:
				mahalan_21_tpr = tpr
				mahalan_21_fpr = fpr
			print('(' + str(fpr) + ', ' + str(tpr) + ')')
			
			# Representar puntos
		
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
				euclid_1_tpr = tpr
				euclid_1_fpr = fpr
			elif i == 1:
				manhatt_1_tpr = tpr
				manhatt_1_fpr = fpr
			elif i == 2:
				mahalan_1_tpr = tpr
				mahalan_1_fpr = fpr
			elif i == 3:
				euclid_5_tpr = tpr
				euclid_5_fpr = fpr
			elif i == 4:
				manhatt_5_tpr = tpr
				manhatt_5_fpr = fpr
			elif i == 5:
				mahalan_5_tpr = tpr
				mahalan_5_fpr = fpr
			elif i == 6:
				euclid_11_tpr = tpr
				euclid_11_fpr = fpr
			elif i == 7:
				manhatt_11_tpr = tpr
				manhatt_11_fpr = fpr
			elif i == 8:
				mahalan_11_tpr = tpr
				mahalan_11_fpr = fpr
			elif i == 9:
				euclid_21_tpr = tpr
				euclid_21_fpr = fpr
			elif i == 10:
				manhatt_21_tpr = tpr
				manhatt_21_fpr = fpr
			elif i == 11:
				mahalan_21_tpr = tpr
				mahalan_21_fpr = fpr
			print('(' + str(fpr) + ', ' + str(tpr) + ')')
	
		# Representar puntos
	
	####################################################################################################################
	
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
			print('(' + str(fpr) + ', ' + str(tpr) + ')')
			# Representar puntos
		
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
			print('(' + str(fpr) + ', ' + str(tpr) + ')')
			# Representar puntos
			

if __name__ == '__main__':
	main()
