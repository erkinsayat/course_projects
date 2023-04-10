import class_model
import matplotlib.pyplot as plt

def predict():
    model = class_model.EmotionClassifier()
    img_for_model = plt.imread('face_screen/face.jpg')
    emotion = model.predict(img_for_model)
    return emotion