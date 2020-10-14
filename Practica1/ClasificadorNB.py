# -*- coding: utf-8 -*-

# coding: utf-8
from abc import ABCMeta, abstractmethod
from scipy import pfd
from Clasificador import Clasificador


class ClasificadorNaiveBayes(Clasificador):
    clase_tabla = []
    clase_probs = []
    
    # TODO: implementar
    def entrenamiento(self, datostrain, atributosDiscretos, diccionario):
        columnas = 3
        clases = len(diccionario["Class"])
        for key in diccionario:
            cols = len(diccionario[key])
            if cols > columnas:
                columnas = cols
        fila = []
        for i in range(0, columnas):
            fila.append(0)
        frecuencias = []
        for i in range(0, len(atributosDiscretos) - 1):
            frecuencias.append(fila)
        for i in range(0, clases):
            self.clase_tabla.append(frecuencias)
            self.clase_probs.append(0)
        for dato in datostrain:
            self.clase_probs[dato[len(atributosDiscretos) - 1]] += 1
            for i in range(0, len(atributosDiscretos) - 1):
                if atributosDiscretos[i] is True:
                    self.clase_tabla[dato[len(atributosDiscretos) - 1]][i][dato[i]] += 1
                elif atributosDiscretos[i] is False:
                    self.clase_tabla[dato[len(atributosDiscretos) - 1]][i][0] += dato[i]
                    self.clase_tabla[dato[len(atributosDiscretos) - 1]][i][1] += 1
                else:
                    raise ValueError("atributosDiscretos no válido.")
        
        for i in range(0, clases):
            if atributosDiscretos[0] is True:
                total = 0
                for data in self.clase_tabla[i][0]:
                    total += data
            elif atributosDiscretos[0] is False:
                total = self.clase_tabla[i][0][1]
            else:
                raise ValueError("atributosDiscretos no válido.")
            for j in range(0, len(self.clase_tabla[i])):
                if atributosDiscretos[j] is True:
                    for data in self.clase_tabla[i][j]:
                        data /= total
                elif atributosDiscretos[j] is False:
                    self.clase_tabla[i][i][0] /= self.clase_tabla[i][i][1]
                else:
                    raise ValueError("atributosDiscretos no válido.")
        
        for dato in datostrain:
            for i in range(0, len(atributosDiscretos) - 1):
                if atributosDiscretos[i] is True:
                    continue
                elif atributosDiscretos[i] is False:
                    self.clase_tabla[dato[len(atributosDiscretos) - 1]][i][2] += \
                        (((dato[i] - self.clase_tabla[dato[len(atributosDiscretos) - 1]][i][0]) ** 2) /
                         self.clase_tabla[dato[len(atributosDiscretos) - 1]][i][1])
                else:
                    raise ValueError("atributosDiscretos no válido.")
        for i in range(0, clases):
            self.clase_probs[i] /= datos.shape[0]
            
    # TODO: implementar
    def clasifica(self, datostest, atributosDiscretos, diccionario):
        
        pred = []
        clases = len(diccionario["Class"])
        for dato in datostest:
            clase = -1
            pr = -1
            for i in range(0, clases):
                prob = self.clase_probs[i]
                for j in range(0, len(atributosDiscretos) - 1):
                    if atributosDiscretos[j] is True:
                        prob *= self.clase_tabla[i][j][dato[j]]
                    elif atributosDiscretos[j] is False:
                        prob *= pdf(dato[j], self.clase_tabla[i][j][0], self.clase_tabla[i][j][2])
                    else:
                        raise ValueError("atributosDiscretos no válido.")
                if prob > pr:
                    pr = prob
                    clase = i
            pred.append(clase)
        return pred
