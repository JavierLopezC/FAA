import pandas as pd
import numpy as np


class Datos:
    # TODO: procesar el fichero para asignar correctamente las variables nominalAtributos, datos y diccionario
    def __init__(self, nombreFichero):
        data = pd.read_csv(nombreFichero, header=0)
        columns = data.columns
        nominalAtributos = []
        diccionario = {}
        for i in range(0, len(columns)-1):
            values = data[columns[i]].unique()
            nominalAtributos.append(False)
            for value in values:
                if not value.isNumeric():
                    if value.isalnum():
                        nominalAtributos[i] = True
                        break
                    else:
                        raise ValueError("Valor no numerico ni nominal.")
            if nominalAtributos[i] is True:
                values = values.sort()
                j = 0
                dic = {}
                for value in values:
                    dic[value] = j
                    j = j+1
            else:
                dic = {}
            diccionario[columns[i]] = dic
        self.datos = data.replace(to_replace=diccionario).to_numpy()
        self.nominalAtributos = nominalAtributos
        self.diccionario = diccionario
        
    # TODO: implementar en la practica 1
    def extraeDatos(self, idx):
        pass
