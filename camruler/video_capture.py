import cv2
import numpy as np
from threading import Thread
import time
import frame_capture


class video_capture(frame_capture.Camera_Thread):
    def __init__(self, path):
        self.cap = cv2.VideoCapture(path)

        # Check if camera opened successfully
        if (self.cap.isOpened() == False):
            print("Unable to read video file")

        # We convert the resolutions from float to integer.
        self.frame_width  = int(self.cap.get(3))
        self.frame_height = int(self.cap.get(4))  
        # self.run()

        # self.view_thread = Thread(target=self.run,args=())
        # self.view_thread.daemon = True
        # self.view_thread.start()
        self.camera_frame_rate = 30
        self.camera_source = path

    def run(self):
        while True:
            _,frame = self.cap.read()
            hsv_frame = cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)

            #Creating a mask
            low_color  = np.array([161, 155, 84])
            high_color = np.array([179, 255, 255])
            
            color_mask = cv2.inRange(hsv_frame,low_color,high_color)

            _, contours, _ = cv2.findContours(color_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            contours = sorted(contours, key=lambda x:cv2.contourArea(x), reverse=True)

            ax = False
            x = 0
            y = 0
            w = 0
            h = 0
            for cnt in contours:
                (x, y, w, h) = cv2.boundingRect(cnt)
                # x_medium = int((x + x + w) / 2)
                ax = True
                break

            if ax:
                cv2.rectangle(frame,(x,y),(x + w, y + h),(0,0,255),2)
                # cv2.line(frame, (x_medium, 0), (x_medium, 480), (0, 255, 0), 2)
            
            self.goal_x = x + w/2
            self.goal_y = y + w/2

            # # cv2.rectangle(frame,(left,top),(right,down),(255,0,0),2)
            # color = (255,125,0)
            # cv2.line(frame, (0,self.top), (480,self.top), (0,0,0), 2)
            # cv2.line(frame, (0,self.down), (480,self.down), color, 2)
            # cv2.line(frame, (self.right, 0), (self.right,480), color, 2)
            # cv2.line(frame, (self.left,0), (self.left,480), color, 2)
            # self.newSP()

            cv2.imshow("Frame",frame) # cv2.imshow("mask",red_mask)

            # Press Q on keyboard to  exit
            if cv2.waitKey(25) & 0xFF == ord('q'):
                break
            
            key = cv2.waitKey(1)
            
        
            # # Move servo motor
            # if x_medium < center -30:
            #     position += 1.5
            # elif x_medium > center + 30:
            #     position -= 1.5
                
            # pwm.setServoPosition(0, position)
            
        self.cap.release()
        self.cv2.destroyAllWindows()



