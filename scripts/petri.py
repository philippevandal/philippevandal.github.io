import sys
import cv2
import numpy as np
import os
import time
import RPi.GPIO as GPIO

GPIO.setmode(GPIO.BOARD)
control_pins = [7,11,13,15]

for pin in control_pins:
  GPIO.setup(pin, GPIO.OUT)
  GPIO.output(pin, 0)

halfstep_seq = [
  [1,0,0,0],
  [1,1,0,0],
  [0,1,0,0],
  [0,1,1,0],
  [0,0,1,0],
  [0,0,1,1],
  [0,0,0,1],
  [1,0,0,1]
]

window_name = 'MicroCV'
cv2.namedWindow(window_name)

min_dist = 10
edge_threshold = 120
centre_threshold = 20
min_radius = 5
max_radius = 40
plate = 4 #each plate could have a tuple with the number of detected bacterial colonies

cap = cv2.VideoCapture(0)

dir_path = '/images'
count = 0
for path in os.listdir(dir_path):
    if os.path.isfile(os.path.join(dir_path, path)):
        count += 1

def generate_mask(img, p, min_dist, edge_threshold, centre_threshold, min_radius, max_radius):
    src = cv2.imread(img, cv2.IMREAD_COLOR)
    gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
    gray = cv2.medianBlur(gray, 5)  # TODO: make interactive
    circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, min_dist,
                              param1=edge_threshold, param2=centre_threshold,
                              minRadius=min_radius, maxRadius=max_radius)
    if circles is not None:
        circles = np.uint16(np.around(circles))
        for i in circles[0, :]:
            center = (i[0], i[1])  # circle center
            cv2.circle(src, center, 1, (0, 100, 100), 3)
            radius = i[2]  # circle outline
            #mask = np.zeros_like(src)
            #mask = cv2.circle(mask, center, radius, (255,255,255), -1)
            cv2.circle(src, center, radius, (255, 0, 255), 3)
            # put mask into alpha channel of input
            result = cv2.cvtColor(src, cv2.COLOR_BGR2BGRA)
            result[:, :, 3] = mask[:,:,0]
            #cv2.imwrite(f'plate_{p}_{count}.png', result)
            count += 1
            print(i)
    return src

def move():
    for i in range(512):
        for halfstep in range(8):
            for pin in range(4):
                GPIO.output(control_pins[pin], halfstep_seq[halfstep][pin])
                time.sleep(0.001)
    GPIO.cleanup()

def grab():
    for i in range(512):
        for halfstep in range(8):
            for pin in range(4):
                GPIO.output(control_pins[pin], halfstep_seq[halfstep][pin])
                time.sleep(0.001)
    GPIO.cleanup()

def motorMove():
    for i in range(512):
        for halfstep in range(8):
            for pin in range(4):
                GPIO.output(control_pins[pin], halfstep_seq[halfstep][pin])
                time.sleep(0.001)
    GPIO.cleanup()

while True:
    ret, frame = cap.read()
    if not ret:
        break
    for i in range(plate):
        #motorMove()
        src = generate_mask(frame, plate, min_dist, edge_threshold, centre_threshold, min_radius, max_radius)
        cv2.imshow(window_name, src)

    key = cv2.waitKey(1) & 0xFF
    # press 'q' to quit the window
    if key == ord('q'):
        break
cv2.destroyAllWindows()
