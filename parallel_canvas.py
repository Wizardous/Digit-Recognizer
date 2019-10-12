import cv2
import numpy as np
# creating a 600 x 600 pixels canvas for mouse drawing
canvas = np.ones((400,1200), dtype="uint8") * 255
# designating a 400 x 400 pixels point of interest on which digits will be drawn
canvas[0:400,0:400] = 0
canvas[401:800, 0:400] = 150
canvas[801:1200, 0:400] = 0


print("Running Canvas...")

while(True):
    cv2.imshow("Test Canvas", canvas)

cv2.destroyAllWindows()