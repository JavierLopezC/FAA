# -*- coding: utf-8 -*-

# coding: utf-8
from abc import ABCMeta, abstractmethod
from EstrategiaParticionado import ValidacionSimple, ValidacionCruzada
from Datos import Datos
from math import sqrt, exp
from scipy.stats import norm
from scipy.spatial.distance import mahalanobis
import numpy as np
from collections import Counter


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
        if isinstance(particionado, ValidacionSimple):
            for i in range(0, 100):
                particionado.creaParticiones(dataset.datos, seed=seed)
        elif isinstance(particionado, ValidacionCruzada):
            particionado.creaParticiones(dataset.datos, seed=seed)
        particiones = particionado.particiones
        # - Para validacion cruzada: en el bucle hasta nv entrenamos el clasificador con la particion de train i
        # y obtenemos el error en la particion de test i
        i = 0
        totalErr = []
        for particion in particiones:
            error = []
            clasificador.entrenamiento(dataset.extraeDatos(particion.indicesTrain),
                                       dataset.nominalAtributos, dataset.diccionario)
            pred = clasificador.clasifica(dataset.extraeDatos(particion.indicesTest),
                                          dataset.nominalAtributos, dataset.diccionario)
            for prediction in pred:
                error.append(Clasificador.error(dataset.extraeDatos(particion.indicesTest), prediction))
            print("Error en partición " + str(i) + str(error))
            if len(totalErr) == 0:
                totalErr = error[:]
            else:
                for j in range(0, len(totalErr)):
                    totalErr[j] += error[j]
            i += 1
        for j in range(0, len(totalErr)):
            totalErr[j] /= i
        return totalErr
                
        # - Para validacion simple (hold-out): entrenamos el clasificador con la particion de train
        # y obtenemos el error en la particion test. Otra opción es repetir la validación simple un número especificado
        # de veces, obteniendo en cada una un error. Finalmente se calcularía la media.
        
        #elif isinstance(particionado, ValidacionSimple):
        #    clasificador.entrenamiento(dataset.extraeDatos(particiones[0].indicesTrain),
        #                               dataset.nominalAtributos, dataset.diccionario)
        #    pred = clasificador.clasifica(dataset.extraeDatos(particiones[0].indicesTest), dataset.nominalAtributos,
        #                                  dataset.diccionario)
        #    error = Clasificador.error(dataset.extraeDatos(particiones[0].indicesTest), pred[0])
        #    error_lap = Clasificador.error(dataset.extraeDatos(particiones[0].indicesTest), pred[1])
        #    return error, error_lap
        #else:
        #    raise ValueError("Particionado no válido.")
     
        
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
            self.clase_probs[int(dato[len(atributosDiscretos) - 1])] += 1
            for i in range(0, len(atributosDiscretos) - 1):
                if atributosDiscretos[i] is True:
                    self.clase_tabla[int(dato[len(atributosDiscretos) - 1])][i][dato[i]] += 1
                    self.clase_tabla_lap[int(dato[len(atributosDiscretos) - 1])][i][dato[i]] += 1
                elif atributosDiscretos[i] is False:
                    self.clase_tabla[int(dato[len(atributosDiscretos) - 1])][i][0] += dato[i]
                    self.clase_tabla[int(dato[len(atributosDiscretos) - 1])][i][1] += 1
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
                    self.clase_tabla[int(dato[len(atributosDiscretos) - 1])][i][2] += \
                        (((dato[i] - self.clase_tabla[int(dato[len(atributosDiscretos) - 1])][i][0]) ** 2) /
                         self.clase_tabla[int(dato[len(atributosDiscretos) - 1])][i][1])
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
        ret = np.asarray([pred, pred_lap])
        return ret


########################################################################################################################
class ClasificadorVecinosProximos(Clasificador):
    datos = []
    
    @staticmethod
    def calcularMediasDesv(datos, nominalAtributos):
        media = []
        desv = []
        for i in range(0, len(nominalAtributos) - 1):
            if nominalAtributos[i] is False:
                media.append(0)
                desv.append(0)
        for dato in datos:
            j = 0
            for i in range(0, len(nominalAtributos) - 1):
                if nominalAtributos[i] is False:
                    media[j] += dato[i]
                    j += 1
        for i in range(0, len(media)):
            media[i] /= len(datos)
        for dato in datos:
            j = 0
            for i in range(0, len(nominalAtributos) - 1):
                if nominalAtributos[i] is False:
                    desv[j] += (dato[i] - media[j]) ** 2
                    j += 1
        for i in range(0, len(desv)):
            desv[i] /= len(datos)
            desv[i] = sqrt(desv[i])
        return media, desv
        
    def normalizarDatos(self, datos, nominalAtributos):
        media, desv = self.calcularMediasDesv(datos, nominalAtributos)
        dato = []
        datos_norm = []
        for i in range(0, len(datos)):
            datos_norm.append([])
        for i in range(0, len(media)):
            dato.append(0)
        for i in range(0, len(datos)):
            k = 0
            for j in range(0, len(nominalAtributos) - 1):
                if nominalAtributos[j] is False:
                    dato[k] = (datos[i][j] - media[k])/desv[k]
                    k += 1
            dato.append(datos[len(nominalAtributos) - 1])
            datos_norm[i] = dato[:]
        return np.array(datos_norm)
        
    # TODO: implementar
    def entrenamiento(self, datostrain, atributosDiscretos, diccionario=None):
        self.datos = self.normalizarDatos(datostrain, atributosDiscretos)
    
    # TODO: implementar
    def clasifica(self, datostest, atributosDiscretos, diccionario=None):
        resultado = [[], [], [], [], [], [], [], [], [], [], [], []]
        datos = self.normalizarDatos(datostest, atributosDiscretos)
        euclid = []
        manhatt = []
        mahalan = []
        for i in range(0, len(datos)):
            euclid.append([])
            manhatt.append([])
            mahalan.append([])
            for trained in self.datos:
                sum_euclid = 0
                sum_manhatt = 0
                for j in range(0, len(trained) - 1):
                    sum_euclid += (datos[i][j] - trained[j]) ** 2
                    sum_manhatt += abs(datos[i][j] - trained[j])
                euclid[i].append(sqrt(sum_euclid))
                manhatt[i].append(sum_manhatt)
                cov = np.cov(np.stack((datos[i], trained), axis=-1))
                inv_cov = np.linalg.inv(cov)
                mahalan[i].append(mahalanobis(datos[i][:-1], trained[:-1], inv_cov))
        for i in range(0, len(datos)):
            euclid_ind = np.argsort(euclid[i])
            manhatt_ind = np.argsort(manhatt[i])
            mahalan_ind = np.argsort(mahalan[i])
            resultado[0].append(self.datos[euclid_ind[0]][-1])
            resultado[1].append(self.datos[manhatt_ind[0]][-1])
            resultado[2].append(self.datos[mahalan_ind[0]][-1])
            
            euclid_list = []
            manhatt_list = []
            mahalan_list = []
            for j in range(0, 5):
                euclid_list.append(self.datos[euclid_ind[j]][-1])
                manhatt_list.append(self.datos[manhatt_ind[j]][-1])
                mahalan_list.append(self.datos[mahalan_ind[j]][-1])
                resultado[3].append(Counter(euclid_list).most_common(1)[0][0])
                resultado[4].append(Counter(manhatt_list).most_common(1)[0][0])
                resultado[5].append(Counter(mahalan_list).most_common(1)[0][0])
            for j in range(5, 11):
                euclid_list.append(self.datos[euclid_ind[j]][-1])
                manhatt_list.append(self.datos[manhatt_ind[j]][-1])
                mahalan_list.append(self.datos[mahalan_ind[j]][-1])
                resultado[6].append(Counter(euclid_list).most_common(1)[0][0])
                resultado[7].append(Counter(manhatt_list).most_common(1)[0][0])
                resultado[8].append(Counter(mahalan_list).most_common(1)[0][0])
            for j in range(11, 21):
                euclid_list.append(self.datos[euclid_ind[j]][-1])
                manhatt_list.append(self.datos[manhatt_ind[j]][-1])
                mahalan_list.append(self.datos[mahalan_ind[j]][-1])
                resultado[9].append(Counter(euclid_list).most_common(1)[0][0])
                resultado[10].append(Counter(manhatt_list).most_common(1)[0][0])
                resultado[11].append(Counter(mahalan_list).most_common(1)[0][0])
        return resultado


########################################################################################################################
class ClasificadorRegresionLogistica(Clasificador):
    w = []
    
    @staticmethod
    def sigmuoidal(x):
        if x < 0:
            return 1 - 1 / (1 + exp(x))
        else:
            return 1 / (1 + exp(-x))
        
    def probClass(self, x):
        prod = 0
        for i in range(0, len(self.w)):
            prod += self.w[i] * x[i]
        return self.sigmuoidal(prod)
        
    def entrenamiento(self, datostrain, atributosDiscretos, diccionario=None):
        w = []
        index_discretos = np.where(atributosDiscretos[:-1] == 1)
        datos = np.delete(datostrain, index_discretos, axis=-1)
        for i in range(0, len(datos[0])):
            w.append(np.random.rand() - 0.5)
        for i in range(0, 100):
            for j in range(0, len(datos)):
                prod = 0
                for k in range(0, len(w)):
                    prod += w[k]*datos[j][k]
                sigma = self.sigmuoidal(prod)
                value_to_vary = 1 * (sigma - datos[j][-1])
                for k in range(0, len(w)):
                    x = value_to_vary * datos[j][k]
                    w[k] -= x
        self.w = w
        
    def clasifica(self, datostest, atributosDiscretos, diccionario=None):
        index_discretos = np.where(atributosDiscretos[:-1] == 1)
        datos = np.delete(datostest, index_discretos, axis=1)
        result = [[]]
        for dato in datos:
            x = dato[:-1]
            x = np.append(x, 1)
            probC1 = self.probClass(x)
            if probC1 >= 0.5:
                result[0].append(1)
            else:
                result[0].append(0)
        return result
