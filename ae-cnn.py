import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from keras.datasets import mnist, fashion_mnist, cifar10 #Carrega as Bases de dados direto do Keras
from keras.models import Model #Classe que permite criar o objeto que vai representar a Rede Neural
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, Dropout, Flatten, Reshape, BatchNormalization, Conv2DTranspose #Classe que permite criar as camadas da rede neural
from keras.optimizers import SGD #Classe que permite trabalhar com o Otmizador alterando seus hiperparâmetros
from keras.utils import to_categorical #Função que permite transformar as saídas de 0, 1, 2 para [0,0,0], [0,1,0], [0,0,1]
from sklearn.metrics import accuracy_score #Funções para calcular a acurácia do modelo

(X_train, y_train), (X_test, y_test) = mnist.load_data()

X_train = X_train.reshape((60000,28,28,1))/255
X_test = X_test.reshape((10000,28,28))/255
 
#ENCODER
inp = Input((28, 28,1))
e = Conv2D(32, (3, 3), activation='relu')(inp)
e = MaxPooling2D((2, 2))(e)
e = Conv2D(64, (3, 3), activation='relu')(e)
e = MaxPooling2D((2, 2))(e)
e = Conv2D(64, (3, 3), activation='relu')(e)
l = Flatten()(e)
l = Dense(49, activation='softmax')(l)

#DECODER
d = Reshape((7,7,1))(l)
d = Conv2DTranspose(64,(3, 3), strides=2, activation='relu', padding='same')(d)
d = BatchNormalization()(d)
d = Conv2DTranspose(64,(3, 3), strides=2, activation='relu', padding='same')(d)
d = BatchNormalization()(d)
d = Conv2DTranspose(32,(3, 3), activation='relu', padding='same')(d)
decoded = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(d)
ae = Model(inp, decoded)
encoder = Model(inp, l)

ae.summary()
ae.compile(optimizer="adam", loss="mse")
relatorio = ae.fit(X_train, X_train, batch_size=512, epochs=30,shuffle=True)

XZ = encoder.predict(X_test)
X_rec = ae.predict(X_test).reshape(10000,28,28)

plt.rcParams['figure.figsize'] = (20,5)
fig, ax = plt.subplots(3, 10)
plt.gray()
for i in np.arange(10):
    ax[0][i].imshow(X_test[i,:,:])
    ax[1][i].imshow(XZ[i,:].reshape(7,7))
    ax[2][i].imshow(X_rec[i,:,:])
plt.show()