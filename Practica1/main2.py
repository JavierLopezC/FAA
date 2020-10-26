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
    
    tictac = Datos("tic-tac-toe.data")
    
    #Analisis ROC
    print("\nValidando 100 veces con clasificador propio:")
    for _ in range(0, 100):
        nb = ClasificadorNaiveBayes()
        vs = ValidacionSimple()
        vs.creaParticiones(tictac.datos)
        particiones = vs.particiones
        nb.entrenamiento(tictac.extraeDatos(particiones[0].indicesTrain),
                                   tictac.nominalAtributos, tictac.diccionario)
        datosTest = tictac.extraeDatos(particiones[0].indicesTest)
        pred, pred_lap = nb.clasifica(datosTest, tictac.nominalAtributos, tictac.diccionario)
        clase = datosTest[:, len(tictac.nominalAtributos) - 1]
        
        tp, tn, fp, fn = valores_roc(clase, pred)
        
        tpr = tp / (tp + fn)
        fpr = fp / (fp + tn)
        print("DatosROC:")
        print(tpr)
        print(fpr)
        print(str(tp) + " " + str(tn) + " " + str(fp) + " " + str(fn))
        #TODO: Añadir el punto (fpr, tpr) en la grafica
    


if __name__ == '__main__':
    main()
