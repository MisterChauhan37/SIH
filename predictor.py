from keras.models import model_from_json
from keras.preprocessing.image import load_img, img_to_array
from keras.optimizers import Adam
import numpy as np


class TDK_model:

    def __init__(self):
        jf = open(r'The_Dark_Knight_architecture2.json', 'r')
        arc = jf.read()
        jf.close()
        self.model = model_from_json(arc)
        self.model.load_weights(r"The_Dark_knight_Weight2.h5")
        adam = Adam(lr=0.01)
        self.model.compile(loss="categorical_crossentropy", optimizer=adam, metrics=["accuracy"])

    def predict(self, path):
        x = img_to_array(load_img(path))
        x = x[np.newaxis, :, :, :]

        pred = self.model.predict(x).tolist()[0]
        l=['Brittle','Ductile','Fatigue','Other']

        return l[pred.index(max(pred))]

if __name__ == "__main__":
	m = TDK_model()
	print(m.predict("C:\\Users\\Harishanker Chauhan\\Desktop\\D.jpg"))