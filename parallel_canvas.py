#Parallel Canvas for 3x Digits

import cv2
import numpy as np
from predict_test import NeuralNet
import os

net = NeuralNet()
class Canvas3:
    def __init__(self):
        # creating a 600 x 600 pixels canv  as for mouse drawing
        self.canvas = np.ones((400,1200), dtype="uint8") * 255
        self.start_point = None
        self.end_point = None
        self.is_drawing = False

    def slice_canvas(self):
        # designating a 400 x 400 pixels point of interest on which digits will be drawn
        self.canvas[0:400, 0:400] = 0
        self.canvas[0:400, 401:800] = 0
        self.canvas[0:400, 801:1200] = 0

    def run_canvs(self):
        self.slice_canvas()
        print("Running Canvas...")
        while(True):
            cv2.imshow("3 Digits Canvas", self.canvas)
            key = cv2.waitKey(1) & 0xFF

            if key == ord('q'):
                break

            elif key == ord('c'):
                self.canvas[0:400, 0:400] = 0
                self.canvas[0:400, 401:800] = 0
                self.canvas[0:400, 801:1200] = 0

            elif key == ord('p'):
                image1, image2, image3 = self.get_slice()
                digit_1 = net.predict(image1)
                digit_2 = net.predict(image2)
                digit_3 = net.predict(image3)
                os.system("cls")
                print("Predicted Number: {}{}{}".format(digit_1, digit_2, digit_3))

        self.close_canvas()

    def draw_line(self, img,start_at,end_at):
        cv2.line(img,start_at,end_at,255,15)

    def on_mouse_events(self,event,x,y,flags,params):
        self.start_point
        self.end_point
        self.is_drawing

        if event == cv2.EVENT_LBUTTONDOWN:
            self.is_drawing = True
            if self.is_drawing:
                self.start_point = (x,y)
        elif event == cv2.EVENT_MOUSEMOVE:
            if self.is_drawing:
                self.end_point = (x,y)
                self.draw_line(self.canvas,self.start_point,self.end_point)
                self.start_point = self.end_point
        elif event == cv2.EVENT_LBUTTONUP:
            self.is_drawing = False

    def close_canvas(self):
        cv2.destroyAllWindows()        

    def get_slice(self):
        slice_1 = self.canvas[0:400, 0:400]
        slice_2 = self.canvas[0:400, 401:800]
        slice_3 = self.canvas[0:400, 801:1200]

        return slice_1, slice_2, slice_3


if __name__ == "__main__":
    C3 = Canvas3()
    cv2.namedWindow("3 Digits Canvas")
    cv2.setMouseCallback("3 Digits Canvas", C3.on_mouse_events)
    C3.run_canvs()
    
