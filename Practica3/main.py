# -*- coding: utf-8 -*-

# coding: utf-8
from Datos import Datos
import numpy as np
from Clasificador import Clasificador, ClasificadorNaiveBayes, ClasificadorGenetico
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
    args = {"epocas": 100, "pob_size": 50}
    prob_cruce = 0
    prob_mutacion = 0
    best_err = 1
    best_args = {}
    for max_reg in range(1, 11):
        while prob_cruce <= 1:
            while prob_mutacion <= 1:
                args["max"] = max_reg
                args["prob_cruce"] = prob_cruce
                args["prob_mutacion"] = prob_mutacion
                gen = ClasificadorGenetico()
                vs = ValidacionSimple()
                error = Clasificador.validacion(vs, titanic, gen, args=args)
                print("Error medio " + str(error[0]))
                if error[0] < best_err:
                    best_err = error[0]
                    best_args = args
                prob_mutacion += 0.05
            prob_cruce += 0.05
    print(str(best_args))

    f = open("./best_args", "w")
    f.write(str(best_args) + "\n" + "Error: " + str(best_err))


    ###################################################################################################################
    #print("\nValidacion Cruzada")
    #titanic = Datos("titanic.data")
    #gen = ClasificadorGenetico()
    #vc = ValidacionCruzada()
    #error = Clasificador.validacion(vc, titanic, gen)
    #print("Error medio " + str(error[0]))


if __name__ == '__main__':
    main()
