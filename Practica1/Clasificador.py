# -*- coding: utf-8 -*-

# coding: utf-8
from abc import ABCMeta, abstractmethod
from EstrategiaParticionado import ValidacionSimple, ValidacionCruzada
from Datos import Datos
from scipy.stats import norm


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
            for particion in particiones:
                clasificador.entrenamiento(dataset.extraeDatos(particion.indicesTrain),
                                           dataset.nominalAtributos, dataset.diccionario)
                pred = clasificador.clasifica(dataset.extraeDatos(particion.indicesTest),
                                              dataset.nominalAtributos, dataset.diccionario)
                error = Clasificador.error(dataset.extraeDatos(particion.indicesTest), pred)
                print("Error en partición " + str(i) + ": " + str(error))
                totalErr += error
                i += 1
            totalErr /= i
            print("Error total: " + str(totalErr))
                
        # - Para validacion simple (hold-out): entrenamos el clasificador con la particion de train
        # y obtenemos el error en la particion test. Otra opción es repetir la validación simple un número especificado
        # de veces, obteniendo en cada una un error. Finalmente se calcularía la media.
        elif isinstance(particionado, ValidacionSimple):
            clasificador.entrenamiento(dataset.extraeDatos(particiones[0].indicesTrain),
                                       dataset.nominalAtributos, dataset.diccionario)

            pred = clasificador.clasifica(dataset.extraeDatos(particiones[0].indicesTest), dataset.nominalAtributos,
                                          dataset.diccionario)
            error = Clasificador.error(dataset.extraeDatos(particiones[0].indicesTest), pred)
            print("Error: " + str(error))
            
        else:
            raise ValueError("Particionado no válido.")
     
        
########################################################################################################################
class ClasificadorNaiveBayes(Clasificador):
    clase_tabla = []
    clase_probs = []
    
    # TODO: implementar
    def entrenamiento(self, datostrain, atributosDiscretos, diccionario):
        self.clase_tabla = []
        self.clase_probs = []
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
            self.clase_probs[i] /= datostrain.shape[0]
    
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
                        prob *= norm.pdf(dato[j], self.clase_tabla[i][j][0], self.clase_tabla[i][j][2])
                    else:
                        raise ValueError("atributosDiscretos no válido.")
                if prob > pr:
                    pr = prob
                    clase = i
            pred.append(clase)
        return pred
