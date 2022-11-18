import cv2
import os

video=cv2.VideoCapture(0)

facedetect=cv2.CascadeClassifier("face_detector\haarcascade_frontalface_default.xml")

count=500

nameID=str(input("Enter Your Name: ")).lower()

path='dataset/'+nameID

isExist = os.path.exists(path)


#os.makedirs(path)

while True:
	ret,frame=video.read()
	faces=facedetect.detectMultiScale(frame,1.3,5)
	for x,y,w,h in faces:
		count=count+1
		name='./dataset/'+nameID+'/'+ str(count) + '.jpg'
		print("Creating Images........." +name)
		cv2.imwrite(name, frame[y:y+h,x:x+w])
		cv2.rectangle(frame, (x,y), (x+w, y+h), (0,255,0), 3)
	cv2.imshow("WindowFrame", frame)
	
	k=cv2.waitKey(1)
	if count >= 1000:

		break
video.release()
cv2.destroyAllWindows()