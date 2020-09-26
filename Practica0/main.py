# -*- coding: utf-8 -*-

# coding: utf-8
from datos import Datos


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
    
    print("\n")

    print("German:\n")
    print("nominalAtributos:")
    print(german.nominalAtributos)
    print("\nDiccionario:")
    print(german.diccionario)
    print("\nDatos:")
    print(german.datos)
    
    
if __name__ == '__main__':
    main()
