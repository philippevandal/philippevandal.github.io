import cv2
import pandas as pd
from time import sleep
import numpy as np
import RPi.GPIO as GPIO

def detect_circles(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    blurred = cv2.GaussianBlur(image, (5, 5), 0)

    # src = cv2.imread(img, cv2.IMREAD_COLOR)
    # gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
    # gray = cv2.medianBlur(gray, 5)
    # circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, min_dist,
    #                           param1=edge_threshold, param2=centre_threshold,
    #                           minRadius=min_radius, maxRadius=max_radius)

    circles = cv2.HoughCircles(blurred, cv2.HOUGH_GRADIENT, dp=1, minDist=min_dist,
                               param1=edge_threshold, param2=centre_threshold, minRadius=min_radius, maxRadius=max_radius)

    if circles is not None:
        circles = circles.round().astype(int)

    return circles

def export_circles_to_csv(circles, iteration, csv_path):
    if circles is None:
        return

    circle_data = []
    for circle in circles[0]:
        x, y, r = circle
        circle_data.append({'x': x, 'y': y, 'radius': r, 'iteration': iteration})

    df = pd.DataFrame(circle_data)

    df.to_csv(csv_path, mode='a', index=False, header=not pd.read_csv(csv_path).empty)

def analyze_iterations(csv_path):
    df = pd.read_csv(csv_path)

    first_iteration_data = df[df['iteration'] == 0]
    second_iteration_data = df[df['iteration'] == 1]
    third_iteration_data = df[df['iteration'] == 2]

    combined_data = pd.concat([first_iteration_data, second_iteration_data, third_iteration_data], ignore_index=True)

    growth_factor = 0.1
    combined_data['growth_radius'] = combined_data['radius'] * np.exp(growth_factor * combined_data['iteration'])

    img = np.ones((480, 480, 3), dtype=np.float32)
    for ind in combined_data.index:
        i = combined_data['iteration'][ind]
        if i == 0:
            color = (255,0,0)
        elif i == 1:
            color = (0,225,0)
        elif 1 == 2:
            color = (0,0,225)

        cv2.circle(img,(combined_data['x'][ind],combined_data['y'][ind]), combined_data['growth_radius'][ind]*10, color, 0.5)
    cv2.imshow(window_name, img)

def move_stepper_motor(dir, steps):
    for i in range(steps):
        for halfstep in range(8):
            if dir == "r":
                for pin in reversed(range(4)):
                    GPIO.output(control_pins[pin], halfstep_seq[halfstep][pin])
                    sleep(0.001)
            else:
                for pin in range(4):
                    GPIO.output(control_pins[pin], halfstep_seq[halfstep][pin])
                    sleep(0.001)
    GPIO.cleanup()

def limit_switch_done(channel):
    move_stepper_motor("a", 0)
    sleep(5)
    move_stepper_motor("a", 50) #TO BE DETERMINED
    global done
    done = True
    GPIO.cleanup()

GPIO.setmode(GPIO.BOARD)
control_pins = [7,11,13,15]

limit_switch_pin = 35
GPIO.setup(limit_switch_pin, GPIO.IN, pull_up_down=GPIO.PUD_UP)
GPIO.add_event_detect(limit_switch_pin, GPIO.FALLING, callback=limit_switch_done, bouncetime=200)

csv_path = 'circles.csv'
iterations = 3

window_name = 'analysis'
img = np.ones((480, 480, 3), dtype=np.float32)
cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
cv2.imshow(window_name, img)

cap = cv2.VideoCapture(0)
min_dist = 10
edge_threshold = 120
centre_threshold = 20
min_radius = 5
max_radius = 40

move_steps = 300
done = False

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

while True:
    for iteration in range(0, iterations):
        move_stepper_motor("a", move_steps)

        ret, frame = cap.read()
        if not ret:
            break

        circles = detect_circles(frame)

        export_circles_to_csv(circles, iteration, csv_path)

    move_stepper_motor("r", 4 * move_steps)

    if done:
        # plot_circle_data(csv_path)
        analyze_iterations(csv_path)
        sleep(10)
        done = False

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
cv2.destroyAllWindows()