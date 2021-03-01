
from picamera.array import PiRGBArray
from picamera import PiCamera
import time
import cv2
import numpy as np
import gensim

screen_width = 640
screen_height = 480

camera = PiCamera()
camera.resolution = (screen_width, screen_height)
camera.framerate = 32
rawCapture = PiRGBArray(camera, size=(screen_width, screen_height))
time.sleep(0.1)

model = gensim.models.KeyedVectors.load_word2vec_format('model.vec') #fasttext model.vec file

font = cv2.FONT_HERSHEY_SIMPLEX
fontScale = 1
fontColor = (0,255,0)
lineType = 2

resolution = 10
resH = screen_height * resolution
resW = screen_width * resolution

for frame in camera.capture_continuous(rawCapture, format="bgr", use_video_port=True):
	image = frame.array
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	blurred = cv2.GaussianBlur(gray, (9, 9), 0)
	edged = cv2.Canny(blurred, 50, 150)
	lines = cv2.HoughLinesP(edged,1,np.pi/180,10,80,1)
	if (lines is not None):
		for line in lines:
			x1,y1,x2,y2 = line[0]
			# cv2.line(resized,(x1,y1),(x2,y2),(255,255,255),2)
			cv2.line(image,(x1,y1),(x2,y2),(0,255,0),2)
			rX = x1 / (resW)
			rY = y1 / (resH)
			rX2 = x2 / (resW)
			rY2 = y2 / (resH)
			# currentX = rX - previousX
			# currentY = rY - previousY
			vector = np.array([rX, rY, rX2, rY2])
			# similar = model.similar_by_vector([x1, y1],topn=1)
			similar = model.similar_by_vector(vector,topn=1)
			# similar2 = model.similar_by_vector([rX2, rY2],topn=1)
			# words.append([similar[0], x1, y1])
			# words.append([similar2[0], x2, y2])
			# similar = model.similar_by_vector(vector,topn=1)
			# similar = model.similar_by_vector([currentX, currentY],topn=1)
			# if(similar != None):
			# if(similar != None || similar2 != None):
				# for w in range(len(words)):
				#     cv2.putText(img, words[w][0], (words[w][1], words[w][2]), font, fontScale, fontColor, lineType)
				#cv2.putText(image, str(similar[0][0]), (x1, y1), font, fontScale, fontColor, lineType)
	rawCapture.truncate(0)
	cv2.imshow("cv", edged)
	key = cv2.waitKey(1) & 0xFF
	rawCapture.truncate(0)
	if key == ord("q"):
		break
