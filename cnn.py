import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from keras.datasets import mnist, fashion_mnist, cifar10 #Carrega as Bases de dados direto do Keras
from keras.models import Model #Classe que permite criar o objeto que vai representar a Rede Neural
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, Dropout, Flatten #Classe que permite criar as camadas da rede neural
from keras.optimizers import SGD #Classe que permite trabalhar com o Otmizador alterando seus hiperparâmetros
from keras.utils import to_categorical #Função que permite transformar as saídas de 0, 1, 2 para [0,0,0], [0,1,0], [0,0,1]

from sklearn.metrics import accuracy_score #Funções para calcular a acurácia do modelo

(X_train, y_train), (X_test, y_test) = mnist.load_data()

X_train = X_train.reshape((60000,28,28,1))
X_test = X_test.reshape((10000,28,28,1))

camada_entrada = Input(shape=(28,28,1))

conv1 = Conv2D(filters=32, kernel_size=(5,5), activation='relu', padding='same')(camada_entrada)
pooling1 = MaxPooling2D(pool_size=(3,3))(conv1)

conv2 = Conv2D(filters=16, kernel_size=(5,5), activation='relu', padding='same')(pooling1)
pooling2 = MaxPooling2D(pool_size=(2,2))(conv2)

flatten = Flatten()(pooling2)

camada1 = Dense(units=128, activation='relu')(flatten)
camada_saida = Dense(units=10, activation='softmax')(camada1)

cnn = Model(camada_entrada, camada_saida)
cnn.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
cnn.summary()
relatorio = cnn.fit(X_train, to_categorical(y_train), batch_size=512, epochs=30, 
        shuffle=True,
        validation_data=(X_test, to_categorical(y_test)))