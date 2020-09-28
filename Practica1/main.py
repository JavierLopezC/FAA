# -*- coding: utf-8 -*-

# coding: utf-8
from Datos import Datos


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
    
    print("\n Datos 0, 2, 6, 10:")
    print(tictac.extraeDatos([0, 2, 6, 10]))
    
    print("\n")

    print("German:\n")
    print("nominalAtributos:")
    print(german.nominalAtributos)
    print("\nDiccionario:")
    print(german.diccionario)
    print("\nDatos:")
    print(german.datos)

    print("\n Datos 5, 9, 2, 1:")
    print(german.extraeDatos([5, 9, 2, 1]))
    
    
if __name__ == '__main__':
    main()
