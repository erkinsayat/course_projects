import cv2
import predict
import time

faceCascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FPS, 24)


out = cv2.VideoWriter('video/video.avi', cv2.VideoWriter_fourcc(*'XVID'), 5.0, (640, 480))

while True:
    ret, img = cap.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.2,
        minNeighbors=5,
        minSize=(20, 20)
    )

    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x-30, y-60), (x+30 + w, y+30 + h), (255, 0, 0), 2)
        cv2.imwrite('face_screen/face.jpg', img[x - 50: x + w, y: y + 30 + h, :])

        emotion = predict.predict()


        cv2.putText(img, f'{emotion}', (y, x - 100), cv2.FONT_HERSHEY_TRIPLEX, 0.9, (0, 0, 255), 1)

    cv2.imshow("camera", img)
    out.write(img)
    #time.sleep(0.05)


    if cv2.waitKey(10) == 27:  # Esc key
        break

cap.release()
cv2.destroyAllWindows()