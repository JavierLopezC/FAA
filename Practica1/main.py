# -*- coding: utf-8 -*-

# coding: utf-8
from Datos import Datos
import numpy as np
from Clasificador import Clasificador, ClasificadorNaiveBayes
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
    #print("error de: " + str(tot))
    return tot

def main():
    tictac = Datos("tic-tac-toe.data")
    german = Datos("german.data")
    
    print("Tic-tac-toe:\n")
    print("nominalAtributos:")
    print(tictac.nominalAtributos)
    print("\nDiccionario:")
    print(tictac.diccionario)
    print("\nDatos:")
    print(tictac.datos)

    print("\n\nTicTacToe:")
    print("\nValidación Simple")

    print("\nValidando 100 veces con clasificador propio:")
    error = 0
    error_lap = 0
    for _ in range(0, 100):
        nb = ClasificadorNaiveBayes()
        vs = ValidacionSimple()
        ret = Clasificador.validacion(vs, tictac, nb)
        error += ret[0]
        error_lap += ret[1]
    print("Error medio sin laplace " + str(error / 100))
    print("Error medio con laplace " + str(error_lap / 100))

    print("\nValidando 100 veces con MultinomialNB:")
    enc = OneHotEncoder(sparse=False)
    tictac.datos = enc.fit_transform(tictac.datos)
    atr = tictac.datos[:, :len(tictac.nominalAtributos) - 1]
    clase = tictac.datos[:, len(tictac.nominalAtributos) - 1]
    error = 0
    for _ in range(0, 100):
        train_x, test_x, train_y, test_y = train_test_split(atr, clase, test_size=0.1)
        nb = MultinomialNB()
        nb.fit(train_x, train_y)
        res = nb.predict(test_x)
        error += error_sk(test_y, res)
    print("Error medio " + str(error / 100))

    print("\nValidando 100 veces con GaussianNB:")
    enc.inverse_transform(tictac.datos)
    atr = tictac.datos[:, :len(tictac.nominalAtributos) - 1]
    clase = tictac.datos[:, len(tictac.nominalAtributos) - 1]
    error = 0
    for _ in range(0, 100):
        train_x, test_x, train_y, test_y = train_test_split(atr, clase, test_size=0.1)
        nb = GaussianNB()
        nb.fit(train_x, train_y)
        res = nb.predict(test_x)
        error += error_sk(test_y, res)
    print("Error medio " + str(error / 100))


    ###################################################################################################################
    print("\nValidacion Cruzada")
    print("\nValidando con clasificador propio:")
    tictac = Datos("tic-tac-toe.data")
    nb = ClasificadorNaiveBayes()
    vc = ValidacionCruzada()
    error = Clasificador.validacion(vc, tictac, nb)
    print("Error medio sin laplace " + str(error[0]))
    print("Error medio con laplace " + str(error[1]))

    print("\nValidando con MultinomialNB:")
    enc = OneHotEncoder(sparse=False)
    tictac.datos = enc.fit_transform(tictac.datos)
    atr = tictac.datos[:, :len(tictac.nominalAtributos) - 1]
    clase = tictac.datos[:, len(tictac.nominalAtributos) - 1]
    acierto = cross_val_score(MultinomialNB(), atr, clase)
    error = []
    for data in acierto:
        error.append(1 - data)
    print("Error por particiones: ")
    print(error)
    total_err = sum(error) / len(error)
    print("Error medio: " + str(total_err))

    print("\nValidando con GaussianNB:")
    enc.inverse_transform(tictac.datos)
    atr = tictac.datos[:, :len(tictac.nominalAtributos) - 1]
    clase = tictac.datos[:, len(tictac.nominalAtributos) - 1]
    acierto = cross_val_score(GaussianNB(), atr, clase)
    error = []
    for data in acierto:
        error.append(1 - data)
    print("Error por particiones: ")
    print(error)
    total_err = sum(error) / len(error)
    print("Error medio: " + str(total_err))

#######################################################################################################################

    print("\n\nGerman:\n")
    print("nominalAtributos:")
    print(german.nominalAtributos)
    print("\nDiccionario:")
    print(german.diccionario)
    print("\nDatos:")
    print(german.datos)

    print("\n\nGerman:")
    print("\nValidacion Simple")
    print("\nValidando 100 veces con clasificador propio:")
    error = 0
    error_lap = 0
    for _ in range(0, 100):
        nb = ClasificadorNaiveBayes()
        vs = ValidacionSimple()
        ret = Clasificador.validacion(vs, german, nb)
        error += ret[0]
        error_lap += ret[1]
    print("Error medio sin laplace " + str(error / 100))
    print("Error medio con laplace " + str(error_lap / 100))

    print("\nValidando 100 veces con MultinomialNB:")
    #En este caso no pretratamos los datos pues los transformaría todos en datos nominales y en este caso hay datos continuos
    atr = german.datos[:, :len(german.nominalAtributos) - 1]
    clase = german.datos[:, len(german.nominalAtributos) - 1]
    error = 0
    for _ in range(0, 100):
        train_x, test_x, train_y, test_y = train_test_split(atr, clase, test_size=0.1)
        nb = MultinomialNB()
        nb.fit(train_x, train_y)
        res = nb.predict(test_x)
        error += error_sk(test_y, res)
    print("Error medio " + str(error / 100))

    print("\nValidando 100 veces con GaussianNB:")
    atr = german.datos[:, :len(german.nominalAtributos) - 1]
    clase = german.datos[:, len(german.nominalAtributos) - 1]
    error = 0
    for _ in range(0, 100):
        train_x, test_x, train_y, test_y = train_test_split(atr, clase, test_size=0.1)
        nb = GaussianNB()
        nb.fit(train_x, train_y)
        res = nb.predict(test_x)
        error += error_sk(test_y, res)
    print("Error medio " + str(error / 100))


    ###################################################################################################################
    print("\nValidacion Cruzada")
    print("\nValidando con clasificador propio:")
    german = Datos("german.data")
    nb = ClasificadorNaiveBayes()
    vc = ValidacionCruzada()
    error = Clasificador.validacion(vc, german, nb)
    print("Error medio sin laplace " + str(error[0]))
    print("Error medio con laplace " + str(error[1]))

    print("\nValidando con MultinomialNB:")
    #En este caso no pretratamos los datos pues los transformaría todos en datos nominales y en este caso hay datos continuos
    atr = german.datos[:, :len(german.nominalAtributos) - 1]
    clase = german.datos[:, len(german.nominalAtributos) - 1]
    acierto = cross_val_score(MultinomialNB(), atr, clase)
    error = []
    for data in acierto:
        error.append(1 - data)
    print("Error por particiones: ")
    print(error)
    total_err = sum(error) / len(error)
    print("Error medio: " + str(total_err))

    print("\nValidando con GaussianNB:")
    atr = german.datos[:, :len(german.nominalAtributos) - 1]
    clase = german.datos[:, len(german.nominalAtributos) - 1]
    acierto = cross_val_score(GaussianNB(), atr, clase)
    error = []
    for data in acierto:
        error.append(1 - data)
    print("Error por particiones: ")
    print(error)
    total_err = sum(error) / len(error)
    print("Error medio: " + str(total_err))


if __name__ == '__main__':
    main()
