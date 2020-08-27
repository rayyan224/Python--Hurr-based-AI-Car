import cv2


# Car Image
carImage="carImg.jpg"
# har features for a car. Pre trained 
car_classifer= "carimageDataset.xml"
pedistrian_classifer="pedistrianDataset.xml"
video= cv2.VideoCapture("Dashcam.mp4")
# video= cv2.VideoCapture("Pedestrians2.mp4")
# Converts car Image(string) into a matrix 
img = cv2.imread(carImage)

# Convert Image to Grayscale
blackWhiteImg =  cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Classifer. We are using the cascade classifer to identify har objects. 
car_tracker= cv2.CascadeClassifier(car_classifer)
pedistrian_tracker= cv2.CascadeClassifier(pedistrian_classifer)




while True: 
    # Read Frame. Everytime its called cv gets a new fram
    (read_suc, frame) = video.read()
    if read_suc == True: 
        grayScale =  cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
    else:
        break

    # Detect Cars and Pedistrians
    cars = car_tracker.detectMultiScale(grayScale)
    pedistrians= pedistrian_tracker.detectMultiScale(grayScale)
    for (x,y,w,h) in cars:
        cv2.rectangle(frame,(x,y),(x+w,y+h), (0,0,255),2)
    for (x,y,w,h) in pedistrians:
        cv2.rectangle(frame,(x,y),(x+w,y+h), (212, 49, 212),2)
    cv2.imshow("car",frame)
    key = cv2.waitKey(1)

    if key == 81 or key == 113: 
        break

video.release()
