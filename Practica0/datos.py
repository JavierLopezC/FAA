# -*- coding: utf-8 -*-

# coding: utf-8
import pandas as pd
import numpy as np


class Datos:
    # TODO: procesar el fichero para asignar correctamente las variables nominalAtributos, datos y diccionario
    def __init__(self, nombreFichero):
        data = pd.read_csv(nombreFichero, header=0)
        columns = data.columns
        nominalAtributos = []
        diccionario = {}
        for i in range(0, len(columns)):
            values = data[columns[i]]
            if np.issubdtype(values.dtype, np.integer) or np.issubdtype(values.dtype, np.floating):
                nominalAtributos.append(False)
            elif values.dtype == object:
                for value in values:
                    if not isinstance(value, str):
                        raise ValueError("Valor no numerico ni nominal.")
                nominalAtributos.append(True)
            else:
                raise ValueError("Valor no numerico ni nominal.")
                        
            if nominalAtributos[i] is True:
                values = pd.unique(values)
                values.sort()
                j = 0
                dic = {}
                for value in values:
                    dic[value] = j
                    j = j+1
            else:
                dic = {}
            diccionario[columns[i]] = dic
        self.datos = data.replace(to_replace=diccionario).values
        self.nominalAtributos = nominalAtributos
        self.diccionario = diccionario
        
    # TODO: implementar en la practica 1
    def extraeDatos(self, idx):
        pass
