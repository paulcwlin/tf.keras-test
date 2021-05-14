import cv2
import numpy as np

bin_path = r'C:\Users\d19fd\Documents\open_model_zoo\tools\downloader\intel\face-detection-adas-0001\FP16\face-detection-adas-0001.bin'
xml_path = r'C:\Users\d19fd\Documents\open_model_zoo\tools\downloader\intel\face-detection-adas-0001\FP16\face-detection-adas-0001.xml'

net = cv2.dnn.readNet(xml_path, bin_path)

net.setPreferableTarget(cv2.dnn.DNN_TARGET_MYRIAD)

cap = cv2.VideoCapture(0)
#ratio = cap.get(cv2.CAP_PROP_FRAME_WIDTH)/cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
#WIDTH = 480
#HEIGHT = int(WIDTH/ratio)

while True:
    ret, frame = cap.read()
    #frame = cv2.resize(frame, (WIDTH, HEIGHT))
    
    blob = cv2.dnn.blobFromImage(frame, size=(300, 300), ddepth=cv2.CV_8U)
    net.setInput(blob)
    out = net.forward()
    
    #faces, scores = [], []
    for detection in out.reshape(-1, 7):
        confidence = float(detection[2])
        xmin = int(detection[3] * frame.shape[1])
        ymin = int(detection[4] * frame.shape[0])
        xmax = int(detection[5] * frame.shape[1])
        ymax = int(detection[6] * frame.shape[0])

        if confidence > 0.5:
            #faces.append([xmin,ymin,xmax-xmin,ymax-ymin])
            #scores.append(confidence)
            print('c', confidence, 'd[3]', detection[3], 'd[4]', detection[4])
            cv2.rectangle(frame, (xmin,ymin), (xmax, ymax), (0,255,0), 2)
    
    #print('face_scores : ', scores)
    #print('face_boxes : ', faces)
    cv2.imshow('video', frame)
        
    key = cv2.waitKey(1)
    if key == 27 or key == ord('q'):
        cap.release()
        cv2.destroyAllWindows()
        break