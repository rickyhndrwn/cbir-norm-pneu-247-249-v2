from tensorflow.keras.applications.densenet import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
import numpy as np

class FeatureExtractor:
    def __init__(self, model):
        self.model = model

    def extract(self, img):
        img = img.resize((224, 224))
        img = img.convert('RGB')

        x = img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)
        
        feature = self.model.predict(x)[0]
        
        return feature / np.linalg.norm(feature)

