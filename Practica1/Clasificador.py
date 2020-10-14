# -*- coding: utf-8 -*-

# coding: utf-8
from abc import ABCMeta, abstractmethod
from EstrategiaParticionado import EstrategiaParticionado, Particion
from Datos import Datos


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
        print("Error de " + str(err) + ".")
        return err
        
    # Realiza una clasificacion utilizando una estrategia de particionado determinada
    # TODO: implementar esta funcion
    def validacion(self, particionado, dataset, clasificador, seed=None):
        # Creamos las particiones siguiendo la estrategia llamando a particionado.creaParticiones
        particiones = particionado.creaParticiones(dataset, seed=seed)
        # - Para validacion cruzada: en el bucle hasta nv entrenamos el clasificador con la particion de train i
        # y obtenemos el error en la particion de test i
        if isinstance(particionado, ValidacionCruzada):
            i = 0
            for particion in particiones:
                clasificador.entrenamiento(dataset.extraerDatos(particion.indicesTrain),
                                           dataset.nominalAtributos, dataset.diccionario)
                pred = clasificador.clasifica(dataset.extraerDatos(particion.indicesTest),
                                              dataset.nominalAtributos, dataset.diccionario)
                error = self.error(dataset.extraerDatos(particion.indicesTest), pred)
                print("Error en partición " + i + ": " + error)
                i = i + 1
                
        # - Para validacion simple (hold-out): entrenamos el clasificador con la particion de train
        # y obtenemos el error en la particion test. Otra opción es repetir la validación simple un número especificado
        # de veces, obteniendo en cada una un error. Finalmente se calcularía la media.
        elif isinstance(particionado, ValidacionSimple):
            clasificador.entrenamiento(dataset.extraerDatos(particiones[0].indicesTrain),
                                       dataset.nominalAtributos, dataset.diccionario)

            pred = clasificador.clasifica(dataset.extraerDatos(particiones[0].indicesTest))
            error = self.error(dataset.extraerDatos(particiones[0].indicesTest), pred)
            print("Error: " + str(error))
            
        else:
            raise ValueError("Particionado no válido.")
