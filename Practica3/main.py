# -*- coding: utf-8 -*-

# coding: utf-8
from Datos import Datos
import numpy as np
from Clasificador import Clasificador, ClasificadorNaiveBayes
from EstrategiaParticionado import ValidacionSimple, ValidacionCruzada
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.naive_bayes import GaussianNB, MultinomialNB


def main():
    titanic = Datos("titanic.data", allNominal=True)
    
    print("Titanic:\n")
    print("nominalAtributos:")
    print(titanic.nominalAtributos)
    print("\nDiccionario:")
    print(titanic.diccionario)
    print("\nDatos:")
    print(titanic.datos)

    print("\n\nTitanic:")
    print("\nValidación Simple")

    print("\nValidando 100 veces:")
    error = 0
    for _ in range(0, 100):
        gen = ClasificadorGenetico()
        vs = ValidacionSimple()
        ret = Clasificador.validacion(vs, titanic, gen)
        error += ret[0]
    print("Error medio " + str(error / 100))

    ###################################################################################################################
    print("\nValidacion Cruzada")
    titanic = Datos("titanic.data")
    gen = ClasificadorGenetico()
    vc = ValidacionCruzada()
    error = Clasificador.validacion(vc, titanic, gen)
    print("Error medio " + str(error[0]))


if __name__ == '__main__':
    main()
