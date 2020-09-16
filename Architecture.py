from keras.layers import Conv2D, MaxPool2D, GlobalMaxPooling2D, BatchNormalization, Dropout, ReLU, add, Input, \
    concatenate, Dense, ELU
from keras.preprocessing.image import load_img, img_to_array, ImageDataGenerator
from keras.utils import to_categorical
import keras as k
import numpy as np
import os


def save(model):
    model_arc = model.to_json()
    with open("/content/gdrive/My Drive/Datasets/The_Dark_Knight_architecture2.json", "w") as jfile:
        jfile.write(model_arc)
    model.save_weights("/content/gdrive/My Drive/Datasets/The_Dark_knight_Weight2.h5")

def res_same_layer(X, channels):
    X = Dropout(0.001)(X)
    X = BatchNormalization()(X)

    A_conv_11_1 = Conv2D(channels // 4, (1, 1), padding="same")(X)

    A_conv_11_3 = Conv2D(channels // 4, (1, 1), padding="same")(X)
    A_conv_33 = Conv2D(channels // 4, (3, 3), padding="same")(A_conv_11_3)

    A_conv_11_5 = Conv2D(channels // 4, (1, 1), padding="same")(X)
    A_conv_55 = Conv2D(channels // 4, (5, 5), padding="same")(A_conv_11_5)

    A_conv_11_7 = Conv2D(channels // 4, (1, 1), padding="same")(X)
    A_conv_77 = Conv2D(channels // 4, (7, 7), padding="same")(A_conv_11_7)

    A1 = concatenate([A_conv_11_1, A_conv_33, A_conv_55, A_conv_77], axis=3)
    A1 = ELU()(A1)

    A1 = Dropout(0.001)(A1)
    A1 = BatchNormalization()(A1)

    B_conv_11_1 = Conv2D(channels // 4, (1, 1), padding="same")(A1)

    B_conv_11_3 = Conv2D(channels // 4, (1, 1), padding="same")(A1)
    B_conv_33 = Conv2D(channels // 4, (3, 3), padding="same")(B_conv_11_3)

    B_conv_11_5 = Conv2D(channels // 4, (1, 1), padding="same")(A1)
    B_conv_55 = Conv2D(channels // 4, (5, 5), padding="same")(B_conv_11_5)

    B_conv_11_7 = Conv2D(channels // 4, (1, 1), padding="same")(A1)
    B_conv_77 = Conv2D(channels // 4, (7, 7), padding="same")(B_conv_11_7)

    B1 = concatenate([B_conv_11_1, B_conv_33, B_conv_55, B_conv_77], axis=3)
    B1 = add([B1, A1])
    B1 = ELU()(B1)
    return B1


def conv_net(shape):
    X_in = Input(shape)

    X = res_same_layer(X_in, 256)
    X = Dropout(0.001)(X)
    X = BatchNormalization()(X)
    X = Conv2D(200, (3, 3), activation="elu")(X)
    X = MaxPool2D()(X)

    X = res_same_layer(X,
     160)
    X = Dropout(0.001)(X)
    X = BatchNormalization()(X)
    X = Conv2D(80, (3, 3), activation="elu")(X)
    X = MaxPool2D()(X)

    X = res_same_layer(X_in, 40)
    X = Dropout(0.001)(X)
    X = BatchNormalization()(X)
    X = Conv2D(20, (3, 3), activation="elu")(X)
    X = MaxPool2D()(X)

    X = res_same_layer(X, 10)
    X = GlobalMaxPooling2D()(X)
    X_out = Dense(4, activation="softmax")(X)
    return k.Model(inputs=X_in, outputs=X_out)


datagen = ImageDataGenerator(validation_split=0.2)
traingen = datagen.flow_from_directory('/content/gdrive/My Drive/Datasets/arai/new', (150, 150))
validgen = datagen.flow_from_directory('/content/gdrive/My Drive/Datasets/arai/valids', (150, 150))

nn = conv_net((None, None, 3))
adam = k.optimizers.Adam(lr=0.01)
nn.compile(loss="categorical_crossentropy", optimizer=adam, metrics=["accuracy"])
nn.fit_generator(traingen, steps_per_epoch=traingen.n // traingen.batch_size, epochs=30)
print(nn.evaluate_generator(validgen, steps=validgen.n // validgen.batch_size))

print("save?")
q = input()
if q == 'y':
    save(nn)