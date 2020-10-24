# -*- coding: utf-8 -*-

# coding: utf-8
from abc import ABCMeta, abstractmethod
from EstrategiaParticionado import ValidacionSimple, ValidacionCruzada
from Datos import Datos
from scipy.stats import norm
import numpy as np


class Clasificador:
    # Clase abstracta
    __metaclass__ = ABCMeta
    
    # Metodos abstractos que se implementan en casa clasificador concreto
    @abstractmethod
    # TODO: esta funcion debe ser implementada en cada clasificador concreto
    # datosTrain: matriz numpy con los datos de entrenamiento
    # atributosDiscretos: array bool con la indicatriz de los atributos nominales
    # diccionario: array de diccionarios de la estructura Datos utilizados para la codificacion de variables discretas
    def entrenamiento(self, datosTrain, atributosDiscretos, diccionario):
        pass
    
    @abstractmethod
    # TODO: esta funcion debe ser implementada en cada clasificador concreto
    # devuelve un numpy array con las predicciones
    def clasifica(self, datosTest, atributosDiscretos, diccionario):
        pass
    
    @staticmethod
    # Obtiene el numero de aciertos y errores para calcular la tasa de fallo
    # TODO: implementar
    def error(datos, pred):
        # Aqui se compara la prediccion (pred) con las clases reales y se calcula el error
        err = 0
        for i in range(0, len(datos)):
            if datos[i][len(datos[i]) - 1] != pred[i]:
                err += 1
        err /= len(datos)
        return err

    @staticmethod
    # Realiza una clasificacion utilizando una estrategia de particionado determinada
    # TODO: implementar esta funcion
    def validacion(particionado, dataset, clasificador, seed=None):
        # Creamos las particiones siguiendo la estrategia llamando a particionado.creaParticiones
        particionado.creaParticiones(dataset.datos, seed=seed)
        particiones = particionado.particiones
        # - Para validacion cruzada: en el bucle hasta nv entrenamos el clasificador con la particion de train i
        # y obtenemos el error en la particion de test i
        if isinstance(particionado, ValidacionCruzada):
            i = 0
            totalErr = 0
            totalErr_lap = 0
            for particion in particiones:
                clasificador.entrenamiento(dataset.extraeDatos(particion.indicesTrain),
                                           dataset.nominalAtributos, dataset.diccionario)
                pred = clasificador.clasifica(dataset.extraeDatos(particion.indicesTest),
                                              dataset.nominalAtributos, dataset.diccionario)
                error = Clasificador.error(dataset.extraeDatos(particion.indicesTest), pred[0])
                print("Error en partición " + str(i) + " sin Laplace: " + str(error))
                error_lap = Clasificador.error(dataset.extraeDatos(particion.indicesTest), pred[1])
                print("Error en partición " + str(i) + " con Laplace: " + str(error_lap))
                totalErr += error
                totalErr_lap += error_lap
                i += 1
            totalErr /= i
            totalErr_lap /= i
            return totalErr, totalErr_lap
                
        # - Para validacion simple (hold-out): entrenamos el clasificador con la particion de train
        # y obtenemos el error en la particion test. Otra opción es repetir la validación simple un número especificado
        # de veces, obteniendo en cada una un error. Finalmente se calcularía la media.
        elif isinstance(particionado, ValidacionSimple):
            clasificador.entrenamiento(dataset.extraeDatos(particiones[0].indicesTrain),
                                       dataset.nominalAtributos, dataset.diccionario)

            pred = clasificador.clasifica(dataset.extraeDatos(particiones[0].indicesTest), dataset.nominalAtributos,
                                          dataset.diccionario)
            error = Clasificador.error(dataset.extraeDatos(particiones[0].indicesTest), pred[0])
            error_lap = Clasificador.error(dataset.extraeDatos(particiones[0].indicesTest), pred[1])
            return error, error_lap
            
        else:
            raise ValueError("Particionado no válido.")
     
        
########################################################################################################################
class ClasificadorNaiveBayes(Clasificador):
    clase_tabla = []
    clase_probs = []
    clase_tabla_lap = []
    clase_probs_lap = []
    
    # TODO: implementar
    def entrenamiento(self, datostrain, atributosDiscretos, diccionario):
        self.clase_tabla = []
        self.clase_probs = []
        self.clase_tabla_lap = []
        self.clase_probs_lap = []
        columnas = 3
        clases = len(diccionario["Class"])
        laplace_tot = 0 - clases
        for key in diccionario:
            cols = len(diccionario[key])
            laplace_tot += cols
            if cols > columnas:
                columnas = cols

        for i in range(0, clases):
            self.clase_tabla.append([])
            self.clase_probs.append(0)
            for j in range(0, len(atributosDiscretos) - 1):
                self.clase_tabla[i].append([])
                for _ in range(0, columnas):
                    self.clase_tabla[i][j].append(0)

        for i in range(0, clases):
            self.clase_tabla_lap.append([])
            for j in range(0, len(atributosDiscretos) - 1):
                self.clase_tabla_lap[i].append([])
                for _ in range(0, columnas):
                    self.clase_tabla_lap[i][j].append(1)

        for dato in datostrain:
            self.clase_probs[dato[len(atributosDiscretos) - 1]] += 1
            for i in range(0, len(atributosDiscretos) - 1):
                if atributosDiscretos[i] is True:
                    self.clase_tabla[dato[len(atributosDiscretos) - 1]][i][dato[i]] += 1
                    self.clase_tabla_lap[dato[len(atributosDiscretos) - 1]][i][dato[i]] += 1
                elif atributosDiscretos[i] is False:
                    self.clase_tabla[dato[len(atributosDiscretos) - 1]][i][0] += dato[i]
                    self.clase_tabla[dato[len(atributosDiscretos) - 1]][i][1] += 1
                else:
                    raise ValueError("atributosDiscretos no válido.")

        self.clase_probs_lap = self.clase_probs[:]
        for prob in self.clase_probs_lap:
            prob += laplace_tot
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
                    k = 0
                    for _ in self.clase_tabla[i][j]:
                        self.clase_tabla[i][j][k] /= total
                        k += 1
                    k = 0
                    for _ in self.clase_tabla_lap[i][j]:
                        self.clase_tabla_lap[i][j][k] /= (total + laplace_tot)
                        k += 1
                elif atributosDiscretos[j] is False:
                    self.clase_tabla[i][j][0] /= self.clase_tabla[i][j][1]
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
            self.clase_probs[i] /= datostrain.shape[0]
        for i in range(0, clases):
            self.clase_probs_lap[i] /= (datostrain.shape[0] + clases * laplace_tot)
    
    # TODO: implementar
    def clasifica(self, datostest, atributosDiscretos, diccionario):
        
        pred = []
        pred_lap = []
        clases = len(diccionario["Class"])
        for dato in datostest:
            clase = -1
            clase_lap = -1
            pr = -1
            pr_lap = -1
            for i in range(0, clases):
                prob = self.clase_probs[i]
                prob_lap = self.clase_probs_lap[i]
                for j in range(0, len(atributosDiscretos) - 1):
                    if atributosDiscretos[j] is True:
                        prob *= self.clase_tabla[i][j][dato[j]]
                        prob_lap *= self.clase_tabla_lap[i][j][dato[j]]
                    elif atributosDiscretos[j] is False:
                        prob *= norm.pdf(dato[j], self.clase_tabla[i][j][0], self.clase_tabla[i][j][2])
                        prob_lap *= norm.pdf(dato[j], self.clase_tabla[i][j][0], self.clase_tabla[i][j][2])
                    else:
                        raise ValueError("atributosDiscretos no válido.")
                if prob > pr:
                    pr = prob
                    clase = i
                if prob_lap > pr_lap:
                    pr_lap = prob_lap
                    clase_lap = i
            pred.append(clase)
            pred_lap.append(clase_lap)
        ret = np.asarray([pred,pred_lap])
        return ret
