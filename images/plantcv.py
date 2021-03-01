# !/usr/bin/python
import sys, traceback
import cv2
import numpy as np
from plantcv import plantcv as pcv

# added librairies
import time
# import itertools Python 3 has zip already as an iterator
import RPi.GPIO as GPIO
import picamera

# OpenCV setup
cv2.namedWindow("plantcv", cv2.WND_PROP_FULLSCREEN)
cv2.setWindowProperty("plantcv", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

# screen = np.zeros((480,640,3), np.uint8)

font                   = cv2.FONT_HERSHEY_SIMPLEX
fontScale              = 1
fontColor              = (255,255,255)
lineType               = 2

#Motors and Leds GPIOs setup
GPIO.setmode(GPIO.BCM)
GPIO.setwarnings(False)
segAnglePins = [5,6,13,19] #will have to check the right GPIOs for the project
leafTangPins = [12,16,20,21] #will have to check the right GPIOs for the project
ledPins = [4, 17]

for (pin1, pin2) in list(zip(segAnglePins, leafTangPins)):
  GPIO.setup(pin1, GPIO.OUT, initial=GPIO.LOW)
  GPIO.setup(pin2, GPIO.OUT, initial=GPIO.LOW)

for pin in ledPins:
    GPIO.setup(pin, GPIO.OUT, initial=GPIO.LOW)

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

# Function for motor movements based on plantcv's algo angles detection
def segmentAngleMotor(angle, motor):
    steps = angle * 4096 / 360
    counter = 0;
        for i in range(512):
            for halfstep in range(8):
                if counter > abs(steps):
                    break
                for pin in range(4):
                    if (angle < 0):
                        CCW = list(reversed(motor))
                        GPIO.output(CCW[pin], halfstep_seq[halfstep][pin])
                    else:
                        GPIO.output(motor[pin], halfstep_seq[halfstep][pin])
                time.sleep(0.001)
                counter += 1
            else:
                continue
            break
    GPIO.cleanup()

while True:
    # Picamera directly to openCV numpy.array
    with picamera.PiCamera() as camera:
        camera.resolution = (320, 240)
        camera.framerate = 24
        time.sleep(2)
        image = np.empty((240, 320, 3), dtype=np.uint8)
        camera.capture(image, 'rgb')

    # Create masked image from a color image based LAB color-space and threshold values.
    # for lower and upper_thresh list as: thresh = [L_thresh, A_thresh, B_thresh]
    mask, masked_img = pcv.threshold.custom_range(rgb_img=image, lower_thresh=[0,0,158], upper_thresh=[255,255,255], channel='LAB')

    # Crop the mask
    # cropped_mask = mask[1150:1750, 900:1550]

    # Skeletonize the mask
    skeleton = pcv.morphology.skeletonize(mask=cropped_mask)

    # Adjust line thickness with the global line thickness parameter (default = 5),
    # and provide binary mask of the plant for debugging. NOTE: the objects and
    # hierarchies returned will be exactly the same but the debugging image (segmented_img)
    # will look different.
    pcv.params.line_thickness = 3

    # Prune the skeleton
    pruned, seg_img, edge_objects = pcv.morphology.prune(skel_img=skeleton, size=0, mask=cropped_mask)

    # Identify branch points
    branch_pts_mask = pcv.morphology.find_branch_pts(skel_img=skeleton, mask=cropped_mask)

    # Identify tip points
    tip_pts_mask = pcv.morphology.find_tips(skel_img=skeleton, mask=None)

    # Sort segments into leaf objects and stem objects
    leaf_obj, stem_obj = pcv.morphology.segment_sort(skel_img=skeleton, objects=edge_objects, mask=cropped_mask)

    # Identify segments
    segmented_img, labeled_img = pcv.morphology.segment_id(skel_img=skeleton, objects=leaf_obj, mask=cropped_mask)

    # Measure path lengths of segments
    labeled_img2 = pcv.morphology.segment_path_length(segmented_img=segmented_img, objects=leaf_obj)

    # Measure euclidean distance of segments
    labeled_img3 = pcv.morphology.segment_euclidean_length(segmented_img=segmented_img, objects=leaf_obj)

    # Measure curvature of segments
    labeled_img4 = pcv.morphology.segment_curvature(segmented_img=segmented_img, objects=leaf_obj)

    # Measure the angle of segments
    labeled_img5 = pcv.morphology.segment_angle(segmented_img=segmented_img, objects=leaf_obj)
    segment_angles = pcv.outputs.observations['segment_angle']['value']

    # Measure the tangent angles of segments
    labeled_img6 = pcv.morphology.segment_tangent_angle(segmented_img=segmented_img, objects=leaf_obj, size=15)
    leaf_tangent_angles = pcv.outputs.observations['segment_tangent_angle']['value']

    # Measure the leaf insertion angles
    labeled_img7 = pcv.morphology.segment_insertion_angle(skel_img=skeleton, segmented_img=segmented_img, leaf_objects=leaf_obj, stem_objects=stem_obj, size=20)

    # Motorized interpretation of segment and leaf insertion angles
    # and displaying informations on display
    if segment_angles and leaf_tangent_angles:
        for (segment, leaf) in zip(segment_angles, leaf_tangent_angles):
            segment_index = segment_angles.index(segment)
            leaf_index = leaf_tangent_angles.index(leaf)

            # Each leaf objects detected are displayed after each other on top
            # of the pruned plant
            pruned[pruned[:, :, 1:].all(axis=-1)] = 0
            leaf_obj[leaf_index][leaf_obj[leaf_index][:, :, 1:].all(axis=-1)] = 0
            dst = cv2.addWeighted(pruned, 1, leaf_obj[leaf_index], 1, 0)
            cv2.imshow('plantcv', dst)

            # Informations are displayed for each leaf detected plantCV
            segAngleText = 'ID:' + segment_index + ' segment angle: ' + segment + '°'
            cv2.putText(dst, sengAngleText ,(400, 30 + 10 * leaf_index)),font,fontScale,fontColor,lineType)

            # Movement for the first stepper motor representing the
            # segment angle. The LED lights up when angle reached
            segmentAngleMotor(segment, segAnglePins)
            GPIO.output(ledPins[0], GPIO.HIGH)
            time.sleep(3)

            leafTangAngleText = 'ID:' + leaf_index + ' leaf tangent angle: ' + leaf + '°'
            cv2.putText(dst, leafTangAngleText ,(400, 35 + 10 * leaf_index)),font,fontScale,fontColor,lineType)

            # Movement for the second stepper motor representing the
            # tangent leaf angle. The LED lights up when angle reached
            segmentAngleMotor(leaf, leafTangPins)
            GPIO.output(ledPins[1], GPIO.HIGH)
            time.sleep(10)

            # After 10 seconds, both stepper motors are brought back to their
            # default position with LEDs turning off
            segmentAngleMotor(-leaf, leafTangPins)
            GPIO.output(ledPins[1], GPIO.LOW)
            time.sleep(3)
            segmentAngleMotor(-segment, segAnglePins)
            GPIO.output(ledPins[0], GPIO.LOW)
            time.sleep(3)

    else:
        time.sleep(1)

    # A picture is taken and analyzed every minute
    time.sleep(20)

    # GPIOs cleanup and exiting the sketch with "q" (or possibly a button?)
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        GPIO.cleanup()
        cv2.destroyAllWindows()
    	break
