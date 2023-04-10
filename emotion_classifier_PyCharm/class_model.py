import numpy as np
from tensorflow.keras.preprocessing import image
import tensorflow as tf

class EmotionClassifier():

    emotion_translate = ['anger', 'contempt', 'disgust',
                         'fear', 'happy', 'neutral',
                         'sad', 'surprise', 'uncertain']

    interpreter = tf.lite.Interpreter(model_path='emotion_classifier.tflite')
    interpreter.allocate_tensors()
    input_index = interpreter.get_input_details()[0]["index"]
    output_index = interpreter.get_output_details()[0]["index"]


    def preprocess_input_facenet(self, image_):
        img = np.copy(image_)
        img = image.img_to_array(img)
        img = np.expand_dims(img, axis=0)
        img = img[..., :: - 1]
        img[..., 0] -= 91.4953
        img[..., 1] -= 103.8827
        img[..., 2] -= 131.0912
        return img

    def predict(self, image_):
        image_ = tf.image.resize(image_, (224, 224))
        image_ = self.preprocess_input_facenet(image_)
        interpreter = self.interpreter


        interpreter.set_tensor(self.input_index, image_.astype(np.float32))
        interpreter.invoke()
        pred = np.argmax(interpreter.get_tensor(self.output_index))
        return self.emotion_translate[pred]
