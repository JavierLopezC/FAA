from abc import ABCMeta,abstractmethod
import math
import random


class Particion():

  # Esta clase mantiene la lista de �ndices de Train y Test para cada partici�n del conjunto de particiones
  def __init__(self):
    self.indicesTrain=[]
    self.indicesTest=[]

#####################################################################################################
class EstrategiaParticionado:
    # Clase abstracta
    __metaclass__ = ABCMeta

    # Atributos: deben rellenarse adecuadamente para cada estrategia concreta. Se pasan en el constructor

    @abstractmethod
    # TODO: esta funcion deben ser implementadas en cada estrategia concreta
    def creaParticiones(self, datos, seed=None):
        pass


#####################################################################################################

class ValidacionSimple(EstrategiaParticionado):

    def __init__(self, propTest=0.9):
        self.tipoEstrategia = "ValidacionSimple"
        self.proporcionTest = propTest
        self.particiones = []

    # Crea particiones segun el metodo tradicional de division de los datos segun el porcentaje deseado y el n�mero de ejecuciones deseado
    # Devuelve una lista de particiones (clase Particion)
    # TODO: implementar

    def creaParticiones(self, datos, seed=None):
        random.seed(seed)
        nRows = datos.shape[0]
        particion = Particion()

        #crea lista de indices del tamaño de datos y realiza la permutación aleatoria
        indexes = list(range(0, nRows))
        random.shuffle(indexes)

        #crea la particion donde corresponde segun especifica la proporcion de test
        numTrain = int(math.ceil(nRows * self.proporcionTest))
        particion.indicesTrain = indexes[0: numTrain]
        particion.indicesTest = indexes[numTrain + 1:]

        # print("\VALIDACION SIMPLE:\n")
        # print("\nTRAIN:\n")
        # print(particion.indicesTrain)
        # print("\nTEST:\n")
        # print(particion.indicesTest)

        self.particiones.append(particion)


#####################################################################################################


class ValidacionCruzada(EstrategiaParticionado):

    def __init__(self, k=10):
        self.tipoEstrategia = "ValidacionCruzada"
        self.numeroParticiones = k
        self.particiones = []

    # Crea particiones segun el metodo de validacion cruzada.
    # El conjunto de entrenamiento se crea con las nfolds-1 particiones y el de test con la particion restante
    # Esta funcion devuelve una lista de particiones (clase Particion)
    # TODO: implementar

    def creaParticiones(self, datos, seed=None):
        random.seed(seed)
        # Creamos permutacion aleatoria con los indices de  datos
        nRows = datos.shape[0]
        indexes = list(range(0, nRows))
        random.shuffle(indexes)
        
        # Dividimos la lista de indices en k sublistas 
        partSize = nRows // self.numeroParticiones
        SubLists = [indexes[i:i + partSize] for i in range(0, len(indexes), partSize)]

        # Cogemos cada iteración una sublista i para utilizar de test y el resto se toman para train
        test = 0
        # print("\nVALIDACION CRUZADA:\n")
        while test < self.numeroParticiones:
            particion = Particion()
            particion.indicesTest = SubLists[test]
            ListAux = [x for j, x in enumerate(SubLists) if j != test]
            particion.indicesTrain = [x for subAux in ListAux for x in subAux]
            # print("\nITER:" + repr(test))
            # print("\nTEST:\n")
            # print(particion.indicesTest)
            # print("\nTRAIN:\n")
            # print(particion.indicesTrain)
            self.particiones.append(particion)
            test += 1
