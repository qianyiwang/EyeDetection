import cv2
import numpy as np
font = cv2.FONT_HERSHEY_SIMPLEX
# import face and eye classifier xml files
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_eye_tree_eyeglasses.xml')
# implement the camera
cap = cv2.VideoCapture(0)
# cap = cv2.VideoCapture('qianyi_Sub_1_Drive_4.mpg')
windowClose = np.ones((5,5),np.uint8)
windowOpen = np.ones((2,2),np.uint8)
windowErode = np.ones((2,2),np.uint8)
while True:
	ret, img = cap.read()
	# img = img[0:len(img[:,1])/2,len(img[1,:])/2:]
	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	# detect eyes
	eyes = eye_cascade.detectMultiScale(gray)
	# draw rectangle around eyes
	for (ex, ey, ew, eh) in eyes:

		cv2.rectangle(gray, (ex,ey), (ex+ew,ey+eh), (0,255,0), 2)
		#do image processing to get the pupil
		eyeImg_gray = gray[ey:ey+eh, ex:ex+ew]
		eyeImg_color = img[ey:ey+eh, ex:ex+ew]
		edges = cv2.Canny(eyeImg_gray,100,200)
		contours,hierarchy = cv2.findContours(edges, 1, 2)
		cv2.drawContours(eyeImg_color,edges,0,(0,0,255),2)
		# for cnt in contours:
		# 	# perimeter = cv2.arcLength(cnt,True)
		# 	# print perimeter
		# 	rect = cv2.minAreaRect(cnt)
		# 	box = cv2.cv.BoxPoints(rect)
		# 	box = np.int0(box)
		# 	area = cv2.contourArea(box)
		# 	if area>250:
		# 		print area
		# 	cv2.drawContours(eyeImg_color,[box],0,(0,0,255),2)

	# show the screen
	cv2.imshow('img', img)

	# stop by pressing esc
	k = cv2.waitKey(30) & 0xff
	if k==27:
		break

cap.release()
cv2.destroyAllWindows()