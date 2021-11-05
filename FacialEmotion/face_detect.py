import cv2
import numpy as np

conf_threshold = 0.5

modelFile = 'res10_300x300_ssd_iter_140000.caffemodel'
configFile = 'deploy.prototxt.txt'
net = cv2.dnn.readNetFromCaffe(configFile, modelFile)

capture = cv2.VideoCapture(1)

while True:
    ret, frame = capture.read()

    (height, width) = frame.shape[:2]

    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0, (300, 300), (104., 177., 123.))
    net.setInput(blob)
    detections = net.forward()

    for i in range(0, detections.shape[2]):

        confidence = detections[0, 0, i, 2]

        print(confidence)

        if confidence > conf_threshold:
            box = detections[0, 0, i, 3:7] * np.array([width, height, width, height])
            (x, y, w, h) = box.astype('int')
            cv2.rectangle(frame, (x, y), (w, h), (255, 255, 0), 2)
            cv2.putText(frame, 'Acc:' + str(confidence), (x, y - 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 1)

    cv2.imshow('Face detect', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

capture.release()
cv2.destroyAllWindows()
