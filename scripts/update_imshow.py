import cv2
import numpy as np
import random
import time

window_name = 'analysis'
img = np.ones((480, 480, 3), dtype=np.float32)
cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
cv2.imshow(window_name, img)

while True:
    # img = np.ones((480, 480, 3), dtype=np.float32)
    for i in range(20):
        cv2.circle(img, (random.random(), random.random()), (0,255,0), -1)
    cv2.imshow(window_name, img)
    time.sleep(1)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
cv2.destroyAllWindows()