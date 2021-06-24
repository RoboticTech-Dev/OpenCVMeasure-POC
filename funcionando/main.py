import cv2
import os
import numpy as np
import matplotlib.pyplot as plt

class FocusedWindow:
    def __init__(self):
        self.rect_start = np.array([[1200, 95], [1200, 685]])
        self.rect_size = np.array([150, 70])
        self.rect_end = self.rect_start + self.rect_size

        # Background: if no background recorded, start if with zeros, and record it during monitoring
        self.path_up = os.path.join(os.getenv('HOME'), 'temp', 'back_up.png')
        self.path_down = os.path.join(os.getenv('HOME'), 'temp', 'back_down.png')
        self.back_up = cv2.imread(self.path_up) if os.path.exists(self.path_up) \
            else np.zeros([self.rect_size[1], self.rect_size[0], 3], dtype=np.uint8)
        self.back_down = cv2.imread(self.path_down) if os.path.exists(self.path_down) \
            else np.zeros([self.rect_size[1], self.rect_size[0], 3], dtype=np.uint8)

    def splitRegion(self, frame, where='up'):
        # Slice the region in RGB and remove background
        frame0 = frame.copy()
        if where == 'up':
            frame0 = frame0.copy()[self.rect_start[0, 1]:self.rect_end[0, 1],
                     self.rect_start[0, 0]:self.rect_end[0, 0], :]
            back = self.back_up
        elif where == 'down':
            frame0 = frame0.copy()[self.rect_start[1, 1]:self.rect_end[1, 1],
                     self.rect_start[1, 0]:self.rect_end[1, 0], :]
            back = self.back_down

        else:
            print('Wrong location name!')
            return False, frame

        # Remove similarities to background
        # for i in range(0, frame0.shape[0]):
        #     for j in range(0, frame0.shape[1]):
        #         if np.linalg.norm(back[i, j, :] - frame0[i, j, :]) < 90:
        #             frame0[i, j, :] = np.array([0, 0, 0], dtype=np.uint8)

        # Gaussian filter
        frame0 = cv2.GaussianBlur(frame0, (3,3), 1)

        # Convert to Grayscale
        gray = cv2.cvtColor(frame0, cv2.COLOR_BGR2GRAY)
        (T, thresh) = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU | cv2.THRESH_BINARY)
        # cv2.imshow('oi', gray)
        # cv2.imshow('mi', thresh)
        # cv2.waitKey(0)

        # Dilation
        rect_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        dilated = cv2.dilate(thresh.copy(), kernel=rect_kernel, iterations=1)

        # Return the region and a confirmation
        # cv2.imshow('frame', frame0)
        # cv2.imshow('back', back)
        # cv2.imshow('open', dilated)
        # cv2.waitKey(0)
        return True, dilated

    def getBoundaries(self, frame):
        pass
        # Estimate contours
        _, contours, _ = cv2.findContours(frame.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if contours:
            # Find the biggest in terms of area
            biggest_index = np.argmax([cv2.contourArea(c) for c in contours])

            # Get top and bottom coordinates
            _, y, _, h = cv2.boundingRect(contours[biggest_index])


            # cv2.drawContours(frame, contours, biggest_index, 255)
            # cv2.imshow('image', frame)
            # cv2.waitKey(0)

            # Return these values to the user
            return y, y+h, cv2.drawContours(frame, contours, biggest_index, 255)
        else:

            return 0, 0

    def recordBackground(self, frame):
        # Record the section highlighted by the rectangles
        self.back_up = frame.copy()[self.rect_start[0, 1]:self.rect_end[0, 1],
                     self.rect_start[0, 0]:self.rect_end[0, 0], :]
        self.back_down = frame.copy()[self.rect_start[1, 1]:self.rect_end[1, 1],
                     self.rect_start[1, 0]:self.rect_end[1, 0], :]

        # Save it to the defined path to get in the next run
        cv2.imwrite(self.path_up, self.back_up, [int(cv2.IMWRITE_PNG_COMPRESSION), 0])
        cv2.imwrite(self.path_down, self.back_down, [int(cv2.IMWRITE_PNG_COMPRESSION), 0])


if __name__ == '__main__':
    # Input video
    camera = cv2.VideoCapture('/home/grin/Downloads/Videos_gopro_mrs/IMG_8398.mov')

    # Start class that deals with the region of interest
    fw = FocusedWindow()

    # For every frame in the video
    grabbed = True
    while grabbed:
        (grabbed, frame) = camera.read()

        if grabbed:
            # Separate focused filtered window
            converted_up, frame_up = fw.splitRegion(frame, 'up')
            converted_down, frame_down = fw.splitRegion(frame, 'down')

            if  converted_down:
                # Calculate the contours and get boundary pixels
                top_u, bottom_u, measures_u = fw.getBoundaries(frame_up)
                top_d, bottom_d, measures_d = fw.getBoundaries(frame_down)

                # Get the pixel coordinates in the original frame
                interior_up = fw.rect_start[0, 1] + bottom_u
                interior_down = fw.rect_start[1, 1] + top_d
                measurement_pix = abs(interior_up - interior_down)

                # Multiply by a scale that represents the real measurement
                scale = 0.19532 # [mm/pixel] -> medida real de 2954 em uma imagem de 577 pixels
                measurement_real = int(measurement_pix/scale)

                # Draw the contour analysis
                contour_up = cv2.cvtColor(measures_u, cv2.COLOR_GRAY2BGR)
                contour_down = cv2.cvtColor(measures_d, cv2.COLOR_GRAY2BGR)

                # Show each frame
                original = frame.copy()
                if measurement_real > 2820:
                    frame[fw.rect_start[0, 1]:fw.rect_end[0, 1], fw.rect_start[0, 0]:fw.rect_end[0, 0], :] = contour_up
                    frame[fw.rect_start[1, 1]:fw.rect_end[1, 1], fw.rect_start[1, 0]:fw.rect_end[1, 0], :] = contour_down
                    color = (0, 0, 255) if measurement_real > 3000 or measurement_real < 2890 else (0, 255, 0)
                    cv2.rectangle(frame, tuple(fw.rect_start[0, :]), tuple(fw.rect_end[0, :]),
                                  color=(0, 255, 0), thickness=1)
                    cv2.rectangle(frame, tuple(fw.rect_start[1, :]), tuple(fw.rect_end[1, :]),
                                  color=(0, 255, 0), thickness=1)
                    cv2.putText(frame, org=(800, 850), text='Medida interior: '+str(int(measurement_real)),
                                fontFace=cv2.FONT_HERSHEY_COMPLEX, fontScale=1, color=color, thickness=2)
                    cv2.putText(frame, org=(300, 850), text='Escala: %.2f mm/pixel' % (1/scale),
                                fontFace=cv2.FONT_HERSHEY_COMPLEX, fontScale=1, color=color, thickness=2)

                cv2.imshow('Vagao com medicao', frame)
                if cv2.waitKey(0) == ord('q'):
                    grabbed = False

                if cv2.waitKey(0) == ord('r'):
                    fw.recordBackground(original)

    cv2.destroyAllWindows()