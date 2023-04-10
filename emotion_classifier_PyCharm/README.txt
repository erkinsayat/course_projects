***Проект по детектированию лиц и определению их эмоций***


Ссылки на статьи:
*Модель детектирования в OpenCV
https://xn--90aeniddllys.xn--p1ai/opencv-mashinnoe-zrenie-na-python-poisk-lica-na-foto-chast-4/
*Статья по работе с putText
https://docs.opencv.org/2.4/modules/core/doc/drawing_functions.html#cv2.putText


Структура:

face_screen - папка содержит снимок лица в процессе детектирования который мы подаем модели
для классификации эмоции в данный момент(необходимо создать)

class_model - содержит класс "EmotionClassifier", который в свою очередь содержит
важную для нас функцию "predict" в которую мы подаем изображение

detector - активирует веб-камеру, детектирует лицо на изображении и сохраняет
обрезанное изображение с лицом

emotion_classifier.tflite - модель в сжатом формате tflite содержит архитектуру и веса(модель полученная в результате обучения)

haarcascade_frontalface_default.xml - настройки детектора лиц в OpenCV

predict - содержит функцию которая в свою очередь вызывает модель "EmotionClassifier"
и возвращает тип эмоции

requirments.txt - содержит библиотеки и их версии необходимые для реализации всего проекта
установить их можно введя в консоль команду "pip install -r requirments.txt"





