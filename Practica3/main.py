# -*- coding: utf-8 -*-

# coding: utf-8
from Datos import Datos
import numpy as np
from Clasificador import Clasificador, ClasificadorNaiveBayes, ClasificadorGenetico
from EstrategiaParticionado import ValidacionSimple, ValidacionCruzada
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.naive_bayes import GaussianNB, MultinomialNB
import matplotlib.pyplot as	plt

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
    titanic = Datos("titanic.data", allNominal=True)

    print("Titanic:\n")
    print("nominalAtributos:")
    print(titanic.nominalAtributos)
    print("\nDiccionario:")
    print(titanic.diccionario)
    print("\nDatos:")
    print(titanic.datos)


    ##################################################################################################################
    # HALLAR MEJORES PROBABILIDADES Y TAMAÑO DE REGLA (NO EJECUTAR, sale tamaño de regla 10, y ambas probs 1)
    #args = {"epocas": 100, "pob_size": 50, "max": 3, "prob_cruce": 0.5, "prob_mutacion": 0.1, "plot": False}
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

    #print(str(best_args))

    ###################################################################################################################
    # Representamos la evolución del best fit (usamos probabilidades y tamaño de regla mucho más bajas para reducir tiempo de ejecución)
    args = {"epocas": 100, "pob_size": 50, "max": 3, "prob_cruce": 0.5, "prob_mutacion": 0.3, "plot": True}
    print("\n\nTitanic:")
    print("\nValidación Simple, Población 50, 100 épocas")
    gen = ClasificadorGenetico()
    vs = ValidacionSimple()
    error = Clasificador.validacion(vs, titanic, gen, args=args)
    print("\nError medio: " + str(error[0]) + "\n")

    args["epocas"] = 200
    print("\nValidación Simple, Población 50, 200 épocas")
    gen = ClasificadorGenetico()
    vs = ValidacionSimple()
    error = Clasificador.validacion(vs, titanic, gen, args=args)
    print("\nError medio: " + str(error[0]) + "\n")

    args["epocas"] = 100
    args["pob_size"] = 150
    print("\nValidación Simple, Población 150, 100 épocas")
    gen = ClasificadorGenetico()
    vs = ValidacionSimple()
    error = Clasificador.validacion(vs, titanic, gen, args=args)
    print("\nError medio: " + str(error[0]) + "\n")

    args["epocas"] = 200
    print("\nValidación Simple, Población 150, 200 épocas")
    gen = ClasificadorGenetico()
    vs = ValidacionSimple()
    error = Clasificador.validacion(vs, titanic, gen, args=args)
    print("\nError medio: " + str(error[0]) + "\n")

    ###################################################################################################################
    # Análisis ROC
    args = {"epocas": 100, "pob_size": 50, "max": 3, "prob_cruce": 0.5, "prob_mutacion": 0.3, "plot": False}
    print("50-100")
    gen = ClasificadorGenetico()
    vs = ValidacionSimple()
    vs.creaParticiones(titanic.datos)
    particiones = vs.particiones
    gen.entrenamiento(titanic.extraeDatos(particiones[0].indicesTrain), titanic.nominalAtributos, titanic.diccionario, args)
    datosTest = titanic.extraeDatos(particiones[0].indicesTest)
    result = gen.clasifica(datosTest, titanic.nominalAtributos, titanic.diccionario)
    clase = datosTest[:, len(titanic.nominalAtributos) - 1]
    tp, tn, fp, fn = valores_roc(clase, result[0])
    tpr = tp / (tp + fn)
    fpr = fp / (fp + tn)
    X_50_100 = fpr
    Y_50_100 = tpr

    args["pob_size"] = 150
    print("150-100")
    gen = ClasificadorGenetico()
    vs = ValidacionSimple()
    vs.creaParticiones(titanic.datos)
    particiones = vs.particiones
    gen.entrenamiento(titanic.extraeDatos(particiones[0].indicesTrain), titanic.nominalAtributos, titanic.diccionario,
                      args)
    datosTest = titanic.extraeDatos(particiones[0].indicesTest)
    result = gen.clasifica(datosTest, titanic.nominalAtributos, titanic.diccionario)
    clase = datosTest[:, len(titanic.nominalAtributos) - 1]
    tp, tn, fp, fn = valores_roc(clase, result[0])
    tpr = tp / (tp + fn)
    fpr = fp / (fp + tn)
    X_150_100 = fpr
    Y_150_100 = tpr

    args["pob_size"] = 50
    args["epocas"] = 200
    print("50-200")
    gen = ClasificadorGenetico()
    vs = ValidacionSimple()
    vs.creaParticiones(titanic.datos)
    particiones = vs.particiones
    gen.entrenamiento(titanic.extraeDatos(particiones[0].indicesTrain), titanic.nominalAtributos, titanic.diccionario,
                      args)
    datosTest = titanic.extraeDatos(particiones[0].indicesTest)
    result = gen.clasifica(datosTest, titanic.nominalAtributos, titanic.diccionario)
    clase = datosTest[:, len(titanic.nominalAtributos) - 1]
    tp, tn, fp, fn = valores_roc(clase, result[0])
    tpr = tp / (tp + fn)
    fpr = fp / (fp + tn)
    X_50_200 = fpr
    Y_50_200 = tpr

    args["pob_size"] = 150
    print("150-200")
    gen = ClasificadorGenetico()
    vs = ValidacionSimple()
    vs.creaParticiones(titanic.datos)
    particiones = vs.particiones
    gen.entrenamiento(titanic.extraeDatos(particiones[0].indicesTrain), titanic.nominalAtributos, titanic.diccionario,
                      args)
    datosTest = titanic.extraeDatos(particiones[0].indicesTest)
    result = gen.clasifica(datosTest, titanic.nominalAtributos, titanic.diccionario)
    clase = datosTest[:, len(titanic.nominalAtributos) - 1]
    tp, tn, fp, fn = valores_roc(clase, result[0])
    tpr = tp / (tp + fn)
    fpr = fp / (fp + tn)
    X_150_200 = fpr
    Y_150_200 = tpr

    print("NB")
    nb = ClasificadorNaiveBayes()
    vs = ValidacionSimple()
    vs.creaParticiones(titanic.datos)
    particiones = vs.particiones
    nb.entrenamiento(titanic.extraeDatos(particiones[0].indicesTrain), titanic.nominalAtributos, titanic.diccionario)
    datosTest = titanic.extraeDatos(particiones[0].indicesTest)
    result = nb.clasifica(datosTest, titanic.nominalAtributos, titanic.diccionario)
    clase = datosTest[:, len(titanic.nominalAtributos) - 1]
    tp, tn, fp, fn = valores_roc(clase, result[0])
    tpr = tp / (tp + fn)
    fpr = fp / (fp + tn)
    X_nb = fpr
    Y_nb = tpr

    plt.figure()
    lw = 2
    plt.plot(X_50_100, Y_50_100, "ro", lw=lw, label="Gen 50-100")
    plt.plot(X_150_100, Y_150_100, "go", lw=lw, label="Gen 150-100")
    plt.plot(X_50_200, Y_50_200, "bo", lw=lw, label="Gen 50-200")
    plt.plot(X_150_200, Y_150_200, "yo", lw=lw, label="Gen 150-200")
    plt.plot([0, X_nb, 1], [0, Y_nb, 1], color="deeppink", lw=lw, label="NB")
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("fpr")
    plt.ylabel("tpr")
    plt.title("ROC Titanic")
    plt.legend(loc="lower right")
    plt.show()

########################################################################################################################
# REPETIMOS PARA TICTACTOE

    tictac = Datos("tic-tac-toe.data", allNominal=True)

    print("Tic-tac-toe:\n")
    print("nominalAtributos:")
    print(tictac.nominalAtributos)
    print("\nDiccionario:")
    print(tictac.diccionario)
    print("\nDatos:")
    print(tictac.datos)


    # Representamos la evolución del best fit (usamos probabilidades y tamaño de regla mucho más bajas para reducir tiempo de ejecución)
    args = {"epocas": 100, "pob_size": 50, "max": 3, "prob_cruce": 0.5, "prob_mutacion": 0.3, "plot": True}
    print("\n\nTic-tac-toe:")
    print("\nValidación Simple, Población 50, 100 épocas")
    gen = ClasificadorGenetico()
    vs = ValidacionSimple()
    error = Clasificador.validacion(vs, tictac, gen, args=args)
    print("\nError medio: " + str(error[0]) + "\n")

    args["epocas"] = 200
    print("\nValidación Simple, Población 50, 200 épocas")
    gen = ClasificadorGenetico()
    vs = ValidacionSimple()
    error = Clasificador.validacion(vs, tictac, gen, args=args)
    print("\nError medio: " + str(error[0]) + "\n")

    args["epocas"] = 100
    args["pob_size"] = 150
    print("\nValidación Simple, Población 150, 100 épocas")
    gen = ClasificadorGenetico()
    vs = ValidacionSimple()
    error = Clasificador.validacion(vs, tictac, gen, args=args)
    print("\nError medio: " + str(error[0]) + "\n")

    args["epocas"] = 200
    print("\nValidación Simple, Población 150, 200 épocas")
    gen = ClasificadorGenetico()
    vs = ValidacionSimple()
    error = Clasificador.validacion(vs, tictac, gen, args=args)
    print("\nError medio: " + str(error[0]) + "\n")


    ###################################################################################################################
    # Análisis ROC
    args = {"epocas": 100, "pob_size": 50, "max": 3, "prob_cruce": 0.5, "prob_mutacion": 0.3, "plot": False}
    print("50-100")
    gen = ClasificadorGenetico()
    vs = ValidacionSimple()
    vs.creaParticiones(tictac.datos)
    particiones = vs.particiones
    gen.entrenamiento(tictac.extraeDatos(particiones[0].indicesTrain), tictac.nominalAtributos, tictac.diccionario, args)
    datosTest = tictac.extraeDatos(particiones[0].indicesTest)
    result = gen.clasifica(datosTest, tictac.nominalAtributos, tictac.diccionario)
    clase = datosTest[:, len(tictac.nominalAtributos) - 1]
    tp, tn, fp, fn = valores_roc(clase, result[0])
    tpr = tp / (tp + fn)
    fpr = fp / (fp + tn)
    X_50_100 = fpr
    Y_50_100 = tpr

    args["pob_size"] = 150
    print("150-100")
    gen = ClasificadorGenetico()
    vs = ValidacionSimple()
    vs.creaParticiones(tictac.datos)
    particiones = vs.particiones
    gen.entrenamiento(tictac.extraeDatos(particiones[0].indicesTrain), tictac.nominalAtributos, tictac.diccionario,
                      args)
    datosTest = tictac.extraeDatos(particiones[0].indicesTest)
    result = gen.clasifica(datosTest, tictac.nominalAtributos, tictac.diccionario)
    clase = datosTest[:, len(tictac.nominalAtributos) - 1]
    tp, tn, fp, fn = valores_roc(clase, result[0])
    tpr = tp / (tp + fn)
    fpr = fp / (fp + tn)
    X_150_100 = fpr
    Y_150_100 = tpr

    args["pob_size"] = 50
    args["epocas"] = 200
    print("50-200")
    gen = ClasificadorGenetico()
    vs = ValidacionSimple()
    vs.creaParticiones(tictac.datos)
    particiones = vs.particiones
    gen.entrenamiento(tictac.extraeDatos(particiones[0].indicesTrain), tictac.nominalAtributos, tictac.diccionario,
                      args)
    datosTest = tictac.extraeDatos(particiones[0].indicesTest)
    result = gen.clasifica(datosTest, tictac.nominalAtributos, tictac.diccionario)
    clase = datosTest[:, len(tictac.nominalAtributos) - 1]
    tp, tn, fp, fn = valores_roc(clase, result[0])
    tpr = tp / (tp + fn)
    fpr = fp / (fp + tn)
    X_50_200 = fpr
    Y_50_200 = tpr

    args["pob_size"] = 150
    print("150-200")
    gen = ClasificadorGenetico()
    vs = ValidacionSimple()
    vs.creaParticiones(tictac.datos)
    particiones = vs.particiones
    gen.entrenamiento(tictac.extraeDatos(particiones[0].indicesTrain), tictac.nominalAtributos, tictac.diccionario,
                      args)
    datosTest = tictac.extraeDatos(particiones[0].indicesTest)
    result = gen.clasifica(datosTest, tictac.nominalAtributos, tictac.diccionario)
    clase = datosTest[:, len(tictac.nominalAtributos) - 1]
    tp, tn, fp, fn = valores_roc(clase, result[0])
    tpr = tp / (tp + fn)
    fpr = fp / (fp + tn)
    X_150_200 = fpr
    Y_150_200 = tpr

    print("NB")
    nb = ClasificadorNaiveBayes()
    vs = ValidacionSimple()
    vs.creaParticiones(tictac.datos)
    particiones = vs.particiones
    nb.entrenamiento(tictac.extraeDatos(particiones[0].indicesTrain), tictac.nominalAtributos, tictac.diccionario)
    datosTest = tictac.extraeDatos(particiones[0].indicesTest)
    result = nb.clasifica(datosTest, tictac.nominalAtributos, tictac.diccionario)
    clase = datosTest[:, len(tictac.nominalAtributos) - 1]
    tp, tn, fp, fn = valores_roc(clase, result[0])
    tpr = tp / (tp + fn)
    fpr = fp / (fp + tn)
    X_nb = fpr
    Y_nb = tpr

    plt.figure()
    lw = 2
    plt.plot(X_50_100, Y_50_100, "ro", lw=lw, label="Gen 50-100")
    plt.plot(X_150_100, Y_150_100, "go", lw=lw, label="Gen 150-100")
    plt.plot(X_50_200, Y_50_200, "bo", lw=lw, label="Gen 50-200")
    plt.plot(X_150_200, Y_150_200, "yo", lw=lw, label="Gen 150-200")
    plt.plot([0, X_nb, 1], [0, Y_nb, 1], color="deeppink", lw=lw, label="NB")
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("fpr")
    plt.ylabel("tpr")
    plt.title("ROC Tic-tac-toe")
    plt.legend(loc="lower right")
    plt.show()


if __name__ == '__main__':
    main()
