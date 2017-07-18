import numpy as np
import cv2
import math
import cv

PATH='shoe.jpg'

def edge(image):
	# sigma=0.3	
	# v=np.median(image)
	# lower=int(max(0,(1.0-sigma)*v))
	# upper=int(min(255,(1.0+sigma)*v))
	# edge=cv2.Canny(image,50,150)
	laplacian=cv2.Laplacian(image,cv2.CV_64F)
	sobelX=cv2.Sobel(image,cv2.CV_64F,1,0,ksize=3)
	sobelY=cv2.Sobel(image,cv2.CV_64F,0,1,ksize=3)
	sobel=np.zeros((image.shape[0],image.shape[1],3),np.uint8)
	for i in range(0,image.shape[0]):
		for j in range(0,image.shape[1]):
			for k in range(0,3):
				temp=math.sqrt(sobelX[i,j,k]*sobelX[i,j,k] + sobelY[i,j,k]*sobelY[i,j,k])
				if(temp>255):
					temp=255
				sobel[i,j,k]=temp
	edgeImg=np.max(np.array([sobel[:,:,0],sobel[:,:,1],sobel[:,:,2]]),axis=0)
	mean=np.mean(edgeImg)
	for i in range(0,image.shape[0]):
		for j in range(0,image.shape[1]):
			if(edgeImg[i,j]<=mean):
				edgeImg[i,j]=0
	# cv2.imshow("sobel",edgeImg)
	# cv2.waitKey(0)
	return edgeImg

def findSignficantContours(img,edgeImg):
	contours,heirarchy=cv2.findContours(edgeImg,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
	level1=[]
	for i,tupl in enumerate(heirarchy[0]):
		if(tupl[3]==-1):
			tupl=np.insert(tupl,0,[i])
			level1.append(tupl)
	significant=[]
	toosmall=edgeImg.size*5/100
	for tupl in level1:
		contour=contours[tupl[0]]
		area=cv2.contourArea(contour)
		if area>toosmall:
			significant.append([contour,area])
			cv2.drawContours(img,[contour],0,(0,255,0),2)
	significant.sort(key=lambda x: x[1])
	# cv2.imshow("img",img)
	# cv2.waitKey(0)
	return [x[0] for x in significant]

im=cv2.imread(PATH,1)
img=cv2.GaussianBlur(im,(5,5),0)
img_gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
# cv2.imshow("gray",img)
edgeImg=edge(img)
edgeImg_8u=np.asarray(edgeImg,np.uint8)
significant=findSignficantContours(img,edgeImg_8u)
mask=np.zeros((img.shape[0],img.shape[1],3),np.uint8)
# for cont in significant:
# 	cv2.drawContours(mask,[cont],0,(0,255,0),1)
cv2.fillPoly(mask,significant,(255,255,255))
# cv2.imshow('mask',mask)
mask=np.logical_not(mask)
im[mask]=0
cv2.imshow('original',img)
cv2.imshow('modified',im)
cv2.imwrite('output.png',im)
cv2.waitKey(0)