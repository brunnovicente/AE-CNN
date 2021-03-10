import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from keras.datasets import mnist, fashion_mnist, cifar10 #Carrega as Bases de dados direto do Keras
from keras.models import Sequential #Classe que permite criar o objeto que vai representar a Rede Neural
from keras.layers import Dense, Conv2D, MaxPooling2D, Dropout, Flatten #Classe que permite criar as camadas da rede neural
from keras.optimizers import SGD #Classe que permite trabalhar com o Otmizador alterando seus hiperparâmetros
from keras.utils import to_categorical #Função que permite transformar as saídas de 0, 1, 2 para [0,0,0], [0,1,0], [0,0,1]

from sklearn.metrics import accuracy_score #Funções para calcular a acurácia do modelo

(X_train, y_train), (X_test, y_test) = mnist.load_data()

