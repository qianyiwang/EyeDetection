import cv2
import numpy as np
font = cv2.FONT_HERSHEY_SIMPLEX
eyeCloseFlag = True
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
	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	# detect faces
	faces = face_cascade.detectMultiScale(gray, 1.3, 5)
	# draw rectangle around faces
	for (x, y, w, h) in faces:
		cv2.rectangle(img, (x,y), (x+w,y+h), (255,0,0), 2)
		roi_gray = gray[y:y+h, x:x+w]
		roi_color = img[y:y+h, x:x+w]
		# detect eyes
		eyes = eye_cascade.detectMultiScale(roi_gray)
		# draw rectangle around eyes
		for (ex, ey, ew, eh) in eyes:
			cv2.rectangle(roi_color, (ex,ey), (ex+ew,ey+eh), (0,255,0), 2)
			#do image processing to get the pupil
			eyeImg_gray = roi_gray[ey:ey+eh, ex:ex+ew]
			eyeImg_color = roi_color[ey:ey+eh, ex:ex+ew]
			pupilFrame = cv2.equalizeHist(eyeImg_gray)
			ret, pupilFrame = cv2.threshold(pupilFrame,55,255,cv2.THRESH_BINARY)#50 ..nothin 70 is better
			pupilFrame = cv2.morphologyEx(pupilFrame, cv2.MORPH_CLOSE, windowClose)
			pupilFrame = cv2.morphologyEx(pupilFrame, cv2.MORPH_ERODE, windowErode)
			pupilFrame = cv2.morphologyEx(pupilFrame, cv2.MORPH_OPEN, windowOpen)
			mask = cv2.inRange(pupilFrame, 250, 255)
			res = cv2.bitwise_and(pupilFrame, pupilFrame, mask = mask)
			mask = cv2.dilate(mask, None, iterations=1)
			params = cv2.SimpleBlobDetector_Params()
			params.filterByArea = True
			params.minArea = 30
			params.filterByCircularity = True
			params.minCircularity = 0.1
			params.filterByConvexity = True
			params.minConvexity = 0.5
			params.filterByInertia =True
			params.minInertiaRatio = 0.5
			detector = cv2.SimpleBlobDetector(params)
			reversemask=255-mask
			keypoints = detector.detect(reversemask)
			im_with_keypoints = cv2.drawKeypoints(eyeImg_color, keypoints, np.array([]), (0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
			cv2.imshow("Keypoints", im_with_keypoints)
			

			
	# show the screen
	# cv2.imshow('img', keypoints)
	# cv2.imshow('im_with_blobs',im_with_blobs)
	# stop by pressing esc
	k = cv2.waitKey(30) & 0xff
	if k==27:
		break

cap.release()
cv2.destroyAllWindows()