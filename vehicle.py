import cv2 as cv
import numpy as np

#web camera
# cap = cv.VideoCapture('video.mp4')
cap = cv.VideoCapture(0)
line_position = 305
min_width = 80 
min_height = 80
offset = 6 # allowable error between pixels
count = 0

#algo for detection
algo = cv.createBackgroundSubtractorMOG2()

def rectangel_handle(x,y,w,h):
    x1 = int(w/2)
    y1 = int(h/2)

    cx = x+x1
    cy = y+y1

    return cx,cy

detect_ambulance = []




while True:
    ret , frame1 = cap.read()

    gray = cv.cvtColor(frame1, cv.COLOR_BGR2GRAY)
    blur = cv.GaussianBlur(gray, (3,3), 3 )
    
    #applying

    img_new = algo.apply(blur)
    #structure of the element in a image and combines neighbouring pixels
    dilate = cv.dilate(img_new, np.ones((5,5)))
    #returns the structure created using dilate function, shape to algorithm
    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (5,5))
    #advanced transformations
    dilate_data = cv.morphologyEx(dilate, cv.MORPH_CLOSE, kernel)
    #edges from binary image, analyze the shape, reduce the size for better visibility
    counter,h = cv.findContours(dilate_data, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

    cv.line(frame1, (25,line_position), (1200, line_position), (25, 225, 80), 3 )

    for (i,channel) in enumerate(counter):
        #validating things inside counter
        (x,y,w,h) =cv.boundingRect(channel)
        val_counter =  (w>=min_width) and (h>=min_height)

        if not val_counter:
            continue

        cv.rectangle(frame1, (x,y), (x+w, y+h),(0,0,255),3)

        amb= rectangel_handle(x,y,w,h)
        detect_ambulance.append(amb)
        cv.circle(frame1, amb, 4,(255,0,0),-1)


        for(x,y) in detect_ambulance:
            if y <(line_position+offset) and y>(line_position-offset):
                count+=1

            cv.line(frame1, (25,line_position), (1200, line_position), (0,255,0),3)
            detect_ambulance.remove((x,y))
            print('Ambulance Detected'+str(count))


    # cv.putText(frame1,'AMBULANCE DETECTED: ',str(count),cv.FONT_HERSHEY_SIMPLEX, 2, (0,2,9), 3)




    cv.imshow('Video Org', frame1)
    cv.imshow('VID', dilate_data)

    if cv.waitKey(1) & 0xFF == ord('d') :
        break

cv.destroyAllWindows()

cap.release()