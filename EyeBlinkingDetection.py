import cv2
import numpy as np
import thread
import time

font = cv2.FONT_HERSHEY_SIMPLEX
# import face and eye classifier xml files
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_eye_tree_eyeglasses.xml')
# implement the camera
cap = cv2.VideoCapture('AsharmaMetro_Sub_1_Drive_1.mpg')
# cap = cv2.VideoCapture(0)
windowClose = np.ones((5,5),np.uint8)
windowOpen = np.ones((2,2),np.uint8)
windowErode = np.ones((2,2),np.uint8)
lastEyeArea = 0
lastEyePerimeter = 0
eyeCloseFlag = False
last_eyeCloseFlag = False

Time = ''
millis = 0
EndTime = 0
StartTime = int(round(time.time() * 1000))
eyeCloseTime = 0
EyeCloseTime = 0

timeStemp = 0
last_timeStemp = 0
openCount = 0

def touchEdge(pupilArr, eyeArr):
	for arr in pupilArr:
		if arr[0][0]<5:
			return True
		
	return False

# def record_time():
# 	while True:
# 		# Time = time.ctime(time.time())
# 		timeStemp = int(round(time.time() * 1000))
# 		# print Time, millis
# 		time.sleep(1)

# # start timer in new thread
# thread.start_new_thread( record_time, () )

while True:
	ret, img = cap.read()
	img = img[0:len(img[:,1])/2,len(img[1,:])/2:]
	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	# detect faces
	faces = face_cascade.detectMultiScale(gray, 1.3, 5)
	# draw rectangle around faces
	for (x, y, w, h) in faces:
		timeStemp = int(round(time.time() * 1000))
		cv2.rectangle(img, (x,y), (x+w,y+h), (255,0,0), 2)
		roi_gray = gray[y:y+h, x:x+w]
		roi_color = img[y:y+h, x:x+w]
		# detect eyes
		eyes = eye_cascade.detectMultiScale(roi_gray)
		# draw rectangle around eyes
		for (ex, ey, ew, eh) in eyes:
			if ey+y<y+h/2:
				cv2.rectangle(roi_color, (ex,ey), (ex+ew,ey+eh), (0,255,0), 2)
				#do image processing to get the pupil
				eyeImg_gray = roi_gray[ey:ey+eh, ex:ex+ew]
				eyeImg_color = roi_color[ey:ey+eh, ex:ex+ew]
				# cv2.circle(roi_color,(ex,ey+eh/3),5,(0,0,255),-1)
				pupilFrame = cv2.equalizeHist(eyeImg_gray)
				ret, pupilFrame = cv2.threshold(pupilFrame,55,255,cv2.THRESH_BINARY)#50 ..nothin 70 is better
				pupilFrame = cv2.morphologyEx(pupilFrame, cv2.MORPH_CLOSE, windowClose)
				pupilFrame = cv2.morphologyEx(pupilFrame, cv2.MORPH_ERODE, windowErode)
				pupilFrame = cv2.morphologyEx(pupilFrame, cv2.MORPH_OPEN, windowOpen)
				#now we find the biggest blob and get the centriod
				threshold = cv2.inRange(pupilFrame,250,255) #get the blobs
				contours, hierarchy = cv2.findContours(threshold,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
				if touchEdge(contours[0], eyeImg_color)==False:
					cv2.drawContours(eyeImg_color, contours[0], -1, (0,0,255), 2)
					# print ('eyeImg:',len(eyeImg_color),'contours',contours[0])
					area = cv2.contourArea(contours[0])
					perimeter = cv2.arcLength(contours[0],True)
					if abs(area-lastEyeArea)>350:
						eyeCloseFlag = True
						openCount = 0
						if last_timeStemp!=0 and eyeCloseFlag==last_eyeCloseFlag:
							eyeCloseTime = eyeCloseTime + timeStemp - last_timeStemp
							print 'eye close'
					
					lastEyeArea = area
					lastEyePerimeter = perimeter


		if len(eyes)<2:
			eyeCloseFlag = True
			openCount = 0
			if last_timeStemp!=0 and eyeCloseFlag==last_eyeCloseFlag:
				eyeCloseTime = eyeCloseTime + timeStemp - last_timeStemp
				print 'eye close'
	if eyeCloseFlag == False:
		openCount = openCount+1
		if openCount>3:
			if eyeCloseTime>500:
				EyeCloseTime = EyeCloseTime+eyeCloseTime
			eyeCloseTime = 0
	# else:
	# 	if eyeCloseTime>500:
	# 		EyeCloseTime = EyeCloseTime+eyeCloseTime

	if eyeCloseTime!=0:
		print eyeCloseTime
	cv2.putText(img, str(EyeCloseTime), (10,400), font, 2,(255,0,0),4)
	cv2.putText(img, str(float(int(round(time.time() * 1000)) - StartTime)), (10,450), font, 2,(255,0,0),4)
	
	cv2.putText(img, str(float(EyeCloseTime)/float(int(round(time.time() * 1000)) - StartTime)), (10,500), font, 2,(255,0,0),4)
	if float(int(round(time.time() * 1000)) - StartTime) > 90000:
		break
	last_eyeCloseFlag = eyeCloseFlag
	last_timeStemp = timeStemp
	eyeCloseFlag = False
	# show the screen
	cv2.imshow('img', img)
	# stop by pressing esc
	k = cv2.waitKey(1) & 0xff
	if k==27:
		EndTime = int(round(time.time() * 1000))
		break
print 'total time', EndTime - StartTime
print 'total eye close time', EyeCloseTime
print 'eye close ratio', float(EyeCloseTime)/float(int(round(time.time() * 1000)) - StartTime)
cap.release()
cv2.destroyAllWindows()