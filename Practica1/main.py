# -*- coding: utf-8 -*-

# coding: utf-8
from Datos import Datos
from Clasificador import Clasificador, ClasificadorNaiveBayes
from EstrategiaParticionado import ValidacionSimple, ValidacionCruzada


def main():
    tictac = Datos("tic-tac-toe.data")
    german = Datos("german.data")
    
    print("Tic-tac-toe:\n")
    print("nominalAtributos:")
    print(tictac.nominalAtributos)
    print("\nDiccionario:")
    print(tictac.diccionario)
    print("\nDatos:")
    print(tictac.datos)
    
    nb = ClasificadorNaiveBayes()
    vs = ValidacionSimple()
    Clasificador.validacion(vs, tictac, nb)
    
    nb = ClasificadorNaiveBayes()
    vc = ValidacionCruzada()
    Clasificador.validacion(vc, tictac, nb)
    
    print("\n")

    print("German:\n")
    print("nominalAtributos:")
    print(german.nominalAtributos)
    print("\nDiccionario:")
    print(german.diccionario)
    print("\nDatos:")
    print(german.datos)

    nb = ClasificadorNaiveBayes()
    vs = ValidacionSimple()
    Clasificador.validacion(vs, german, nb)
    
    nb = ClasificadorNaiveBayes()
    vc = ValidacionCruzada()
    Clasificador.validacion(vc, german, nb)
    
    
if __name__ == '__main__':
    main()
