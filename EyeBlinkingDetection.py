import cv2
import numpy as np
font = cv2.FONT_HERSHEY_SIMPLEX
# import face and eye classifier xml files
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_eye_tree_eyeglasses.xml')
# implement the camera
cap = cv2.VideoCapture(0)
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
			pupilO = pupilFrame
			ret, pupilFrame = cv2.threshold(pupilFrame,55,255,cv2.THRESH_BINARY)#50 ..nothin 70 is better
			pupilFrame = cv2.morphologyEx(pupilFrame, cv2.MORPH_CLOSE, windowClose)
			pupilFrame = cv2.morphologyEx(pupilFrame, cv2.MORPH_ERODE, windowErode)
			pupilFrame = cv2.morphologyEx(pupilFrame, cv2.MORPH_OPEN, windowOpen)
			#now we find the biggest blob and get the centriod
			threshold = cv2.inRange(pupilFrame,250,255) #get the blobs
			contours, hierarchy = cv2.findContours(threshold,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)

			maxArea = 0
			for cnt in contours:
				area = cv2.contourArea(cnt)
				if area > maxArea:
					maxArea = area
					largeBlob = cnt
					
			if len(largeBlob) > 1:	
				center = cv2.moments(largeBlob)
				cx,cy = int(center['m10']/center['m00']), int(center['m01']/center['m00'])
				cv2.circle(eyeImg_color,(cx,cy),5,(0,0,255),-1)
				cv2.drawContours(eyeImg_color, largeBlob, -1, (0,0,255), 2)
				# epsilon = 0.1*cv2.arcLength(largeBlob,True)
				# approx = cv2.approxPolyDP(largeBlob,epsilon,True)
				# cv2.drawContours(eyeImg_color, approx, -1, (0,0,255), 2)

	# show the screen
	cv2.imshow('img', img)
	# stop by pressing esc
	k = cv2.waitKey(30) & 0xff
	if k==27:
		break

cap.release()
cv2.destroyAllWindows()