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
    #args = {"epocas": 100, "pob_size": 50, "max": 3, "prob_cruce": 0.5, "prob_mutacion": 0.1}
    #best_err = 1
    #best_args = {}
    #for max_reg in range(1, 11):
    #    args["max"] = max_reg
    #    gen = ClasificadorGenetico()
    #    vs = ValidacionSimple()
    #    error = Clasificador.validacion(vs, titanic, gen, args=args)
    #    if error[0] < best_err:
    #        best_err = error[0]
    #        best_args = args

    #args["max"] = best_args["max"]

    #best_err = 1
    #best_args = {}
    #for prob_cruce in np.arange(0, 1, 0.05):
    #    args["prob_cruce"] = prob_cruce
    #    gen = ClasificadorGenetico()
    #    vs = ValidacionSimple()
    #    error = Clasificador.validacion(vs, titanic, gen, args=args)
    #    if error[0] < best_err:
    #        best_err = error[0]
    #        best_args = args

    #args["prob_cruce"] = best_args["prob_cruce"]
    
    #for prob_mutacion in np.arange(0, 1, 0.05):
    #    args["prob_mutacion"] = prob_mutacion
    #    gen = ClasificadorGenetico()
    #    vs = ValidacionSimple()
    #    error = Clasificador.validacion(vs, titanic, gen, args=args)
    #    # print("Error medio " + str(error[0]))
    #    if error[0] < best_err:
    #        best_err = error[0]
    #        best_args = args
    
    #best_err = 1
    #best_args = {}
    #for prob_cruce in np.arange(0.85, 1, 0.05):
    #    args["prob_cruce"] = prob_cruce
    #    for prob_mutacion in np.arange(0.85, 1, 0.05):
    #        args["prob_mutacion"] = prob_mutacion
    #        gen = ClasificadorGenetico()
    #        vs = ValidacionSimple()
    #        error = Clasificador.validacion(vs, titanic, gen, args=args)
    #        if error[0] < best_err:
    #            best_err = error[0]
    #            best_args = args

    #print(str(best_args))

    ###################################################################################################################
    #print("\nValidacion Cruzada")
    #titanic = Datos("titanic.data")
    #gen = ClasificadorGenetico()
    #vc = ValidacionCruzada()
    #error = Clasificador.validacion(vc, titanic, gen)
    #print("Error medio " + str(error[0]))


if __name__ == '__main__':
    main()
