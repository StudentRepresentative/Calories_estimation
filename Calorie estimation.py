
# coding: utf-8

# In[1]:


import cv2
import numpy as np
#import sys

#density - gram / cm^3
density_dict = { 1:0.609, 2:0.94, 3:0.577, 4:0.641, 5:1.151, 6:0.482, 7:0.513, 8:0.641, 9:0.481, 10:0.641, 11:0.521, 12:0.881, 13:0.228, 14:0.650 }
#kcal
calorie_dict = { 1:52, 2:89, 3:92, 4:41, 5:360, 6:47, 7:40, 8:158, 9:18, 10:16, 11:50, 12:61, 13:31, 14:30 }
#skin of photo to real multiplier
skin_multiplier = 5*2.3

def getCalorie(label, volume): #volume in cm^3
	'''
	Inputs are the volume of the foot item and the label of the food item
	so that the food item can be identified uniquely.
	The calorie content in the given volume of the food item is calculated.
	'''
	calorie = calorie_dict[int(label)]
	if (volume == None):
		return None, None, calorie
	density = density_dict[int(label)]
	mass = volume*density*1.0
	calorie_tot = (calorie/100.0)*mass
	return mass, calorie_tot, calorie #calorie per 100 grams

def getVolume(label, area, skin_area, pix_to_cm_multiplier, fruit_contour):
	'''
	Using callibration techniques, the volume of the food item is calculated using the
	area and contour of the foot item by comparing the foot item to standard geometric shapes
	'''
	area_fruit = (area/skin_area)*skin_multiplier #area in cm^2
	label = int(label)
	volume = 100
	if label == 1 or label == 9 or label == 7 or label == 6 or label==12: #sphere-apple,tomato,orange,kiwi,onion
		radius = np.sqrt(area_fruit/np.pi)
		volume = (4/3)*np.pi*radius*radius*radius
		print (area_fruit, radius, volume, skin_area)
	
	if label == 2 or label == 10 or (label == 4 and area_fruit > 30): #cylinder like banana, cucumber, carrot
		fruit_rect = cv2.minAreaRect(fruit_contour)
		height = max(fruit_rect[1])*pix_to_cm_multiplier
		radius = area_fruit/(2.0*height)
		volume = np.pi*radius*radius*height
		
	if (label==4 and area_fruit < 30) or (label==5) or (label==11): #cheese, carrot, sauce
		volume = area_fruit*0.5 #assuming width = 0.5 cm

	if (label==8) or (label==14) or (label==3) or (label==13):
		volume = None
	
	return volume


# In[6]:



import numpy as np
import cv2
from multiprocessing.pool import ThreadPool


def build_filters():
    filters = []
    ksize = 31
    for theta in np.arange(0, np.pi, np.pi / 8):
        for wav in [ 8.0, 13.0]:
            for ar in [0.8, 2.0]:
                kern = cv2.getGaborKernel((ksize, ksize), 5.0, theta, wav, ar, 0, ktype=cv2.CV_32F)
            filters.append(kern)
            cv2.imshow('filt', filters[9])
            return filters
	
def process_threaded(img, filters, threadn = 8):
    accum = np.zeros_like(img)
    def f(kern):
        return cv2.filter2D(img, cv2.CV_8UC3, kern)
    pool = ThreadPool(processes=threadn)
    for fimg in pool.imap_unordered(f, filters):
        np.maximum(accum, fimg, accum)
    return accum

def EnergySum(img):
	mean, dev = cv2.meanStdDev(img)
	return mean[0][0], dev[0][0]
	
def process(img, filters):
    '''
    Given an image and gabor filters,
    the gabor features of the image are calculated.
    '''
    feature = []
    accum = np.zeros_like(img)
    for kern in filters:
        fimg = cv2.filter2D(img, cv2.CV_8UC3, kern)
        a, b = EnergySum(fimg)
        feature.append(a)
        feature.append(b)
        np.maximum(accum, fimg, accum)
    
    M = max(feature)
    m = min(feature)
    feature = map(lambda x: x * 2, feature)
    feature = (feature - M - m)/(M - m);
    mean=np.mean(feature)
    dev=np.std(feature)
    feature = (feature - mean)/dev;
    return feature


def getTextureFeature(img):
    filters = build_filters()
    gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    res1 = process(gray_image, filters)
    return res1

if __name__ == '__main__':
    import sys
    print (__doc__)
    try: img_fn = sys.argv[1]

    except: img_fn = 'test.JPG'
    img = cv2.imread(img_fn) 
    print (getTextureFeature(img))
    cv2.waitKey()
    cv2.destroyAllWindows()


# In[7]:


import cv2
import math
import sys
import numpy as np

def getColorFeature(img):
	'''
	Computes the color feature vector of the image
	based on HSV histogram
	'''
	img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
	h,s,v = cv2.split(img_hsv)
	
	hsvHist = [[[0 for _ in range(2)] for _ in range(2)] for _ in range(6)]
	
	featurevec = []
	hist = cv2.calcHist([img_hsv], [0, 1, 2], None, [6,2,2], [0, 180, 0, 256, 0, 256])	
	for i in range(6):
		for j in range(2):
			for k in range(2):
				featurevec.append(hist[i][j][k])
	feature = featurevec[1:]	
	M = max(feature)
   	m = min(feature)
    	feature = map(lambda x: x * 2, feature)
    	feature = (feature - M - m)/(M - m);
    	mean=np.mean(feature)
    	dev=np.std(feature)
    	feature = (feature - mean)/dev;

	return feature



if __name__ == '__main__':
	img = cv2.imread(sys.argv[1])
	featureVector = getColorFeature(img)
	print featureVector
	cv2.waitKey(0)
	cv2.destroyAllWindows()
    


# In[8]:


import numpy as np
import cv2
#import sys

def getShapeFeatures(img):
    
    contours, hierarchy = cv2.findContours(img, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    moments = cv2.moments(contours[0])
    hu = cv2.HuMoments(moments)
    feature = []
    for i in hu:
        feature.append(i[0])	
        M = max(feature)
        m = min(feature)
    feature = map(lambda x: x * 2, feature)
    feature = (feature - M - m)/(M - m);
    mean=np.mean(feature)
    dev=np.std(feature)
    feature = (feature - mean)/dev;
    return feature

if __name__ == '__main__':
    img = cv2.imread('sys.argv[1]')
    img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    mask = cv2.inRange(img, 80, 255)
    img1 = cv2.bitwise_and(img, img, mask = mask)
    h = getShapeFeatures(img1)
    print (h)
    cv2.waitKey()
    cv2.destroyAllWindows()


# In[9]:


import cv2
import numpy as np
import sys

def getAreaOfFood(img1):
	img = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY
	img_filt = cv2.medianBlur( img, 5)
	img_th = cv2.adaptiveThreshold(img_filt,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,2)
	contours, hierarchy = cv2.findContours(img_th, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

	
	mask = np.zeros(img.shape, np.uint8)
	largest_areas = sorted(contours, key=cv2.contourArea)
	cv2.drawContours(mask, [largest_areas[-1]], 0, (255,255,255,255), -1)
	img_bigcontour = cv2.bitwise_and(img1,img1,mask = mask)


	hsv_img = cv2.cvtColor(img_bigcontour, cv2.COLOR_BGR2HSV)
	h,s,v = cv2.split(hsv_img)
	mask_plate = cv2.inRange(hsv_img, np.array([0,0,100]), np.array([255,90,255]))
	mask_not_plate = cv2.bitwise_not(mask_plate)
	fruit_skin = cv2.bitwise_and(img_bigcontour,img_bigcontour,mask = mask_not_plate)

	#convert to hsv to detect and remove skin pixels
	hsv_img = cv2.cvtColor(fruit_skin, cv2.COLOR_BGR2HSV)
	skin = cv2.inRange(hsv_img, np.array([0,10,60]), np.array([10,160,255])) #Scalar(0, 10, 60), Scalar(20, 150, 255)
	not_skin = cv2.bitwise_not(skin); #invert skin and black
	fruit = cv2.bitwise_and(fruit_skin,fruit_skin,mask = not_skin) #get only fruit pixels

	fruit_bw = cv2.cvtColor(fruit, cv2.COLOR_BGR2GRAY)
	fruit_bin = cv2.inRange(fruit_bw, 10, 255) #binary of fruit

	#erode before finding contours
	kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
	erode_fruit = cv2.erode(fruit_bin,kernel,iterations = 1)

	#find largest contour since that will be the fruit
	img_th = cv2.adaptiveThreshold(erode_fruit,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,2)
	contours, hierarchy = cv2.findContours(img_th, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
	mask_fruit = np.zeros(fruit_bin.shape, np.uint8)
	largest_areas = sorted(contours, key=cv2.contourArea)
	cv2.drawContours(mask_fruit, [largest_areas[-2]], 0, (255,255,255), -1)
	#dilate now
	kernel2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(13,13))
	mask_fruit2 = cv2.dilate(mask_fruit,kernel2,iterations = 1)
	res = cv2.bitwise_and(fruit_bin,fruit_bin,mask = mask_fruit2)
	fruit_final = cv2.bitwise_and(img1,img1,mask = mask_fruit2)
	#find area of fruit
	img_th = cv2.adaptiveThreshold(mask_fruit2,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,2)
	contours, hierarchy = cv2.findContours(img_th, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
	largest_areas = sorted(contours, key=cv2.contourArea)
	fruit_contour = largest_areas[-2]
	fruit_area = cv2.contourArea(fruit_contour)

	
	#finding the area of skin. find area of biggest contour
	skin2 = skin - mask_fruit2
	#erode before finding contours
	kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
	skin_e = cv2.erode(skin2,kernel,iterations = 1)
	img_th = cv2.adaptiveThreshold(skin_e,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,2)
	contours, hierarchy = cv2.findContours(img_th, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
	mask_skin = np.zeros(skin.shape, np.uint8)
	largest_areas = sorted(contours, key=cv2.contourArea)
	cv2.drawContours(mask_skin, [largest_areas[-2]], 0, (255,255,255), -1)

	skin_rect = cv2.minAreaRect(largest_areas[-2])
	box = cv2.cv.BoxPoints(skin_rect)
	box = np.int0(box)
	mask_skin2 = np.zeros(skin.shape, np.uint8)
	cv2.drawContours(mask_skin2,[box],0,(255,255,255), -1)

	pix_height = max(skin_rect[1])
	pix_to_cm_multiplier = 5.0/pix_height
	skin_area = cv2.contourArea(box)


	return fruit_area, mask_fruit2, fruit_final, skin_area, fruit_contour, pix_to_cm_multiplier


if __name__ == '__main__':
	img1 = cv2.imread(sys.argv[1])
	area, bin_fruit, img_fruit, skin_area, fruit_contour, pix_to_cm_multiplier = getAreaOfFood(img1)

	cv2.waitKey()
	cv2.destroyAllWindows()


# In[10]:


import numpy as np
import cv2
from create_feature import * 
from calorie_calc import *
import csv

svm_params = dict( kernel_type = cv2.SVM_LINEAR, svm_type = cv2.SVM_C_SVC, C=2.67, gamma=5.383 )


def training():
	feature_mat = []
	response = []
	for j in [1,2,3,4,5,6,7,8,9,10,11,12,13,14]:
		for i in range(1,21):
			print ("../Dataset/images/All_Images/"+str(j)+"_"+str(i)+".jpg")
			fea, farea, skinarea, fcont, pix_to_cm = readFeatureImg ("../Dataset/images/All_Images/"+str(j)+"_"+str(i)+".jpg")
			feature_mat.append(fea)
			response.append(float(j))
	trainData = np.float32(feature_mat).reshape(-1,94)
	responses = np.float32(response)
	svm = cv2.SVM()
	svm.train(trainData,responses,params = svm_params)
	svm.save('svm_data.dat')	

def testing():
    svm_model = cv2.SVM()
    svm_model.load('svm_data.dat')
    feature_mat = []
    response = []
    image_names = []
    pix_cm = []
    fruit_contours = []
    fruit_areas = []
    fruit_volumes = []
    fruit_mass = []
    fruit_calories = []
    skin_areas = []
    fruit_calories_100grams = []
    for j in [1,2,3,4,5,6,7,8,9,10,11,12,13,14]:
        for i in range(21,26):
            img_path = "../Dataset/images/Test_Images/"+str(j)+"_"+str(i)+".jpg"
            print(img_path)
            fea, farea, skinarea, fcont, pix_to_cm = readFeatureImg (img_path)
            pix_cm.append(pix_to_cm)
            fruit_contours.append(fcont)
            fruit_areas.append(farea)
            feature_mat.append(fea)
            skin_areas.append(skinarea)
            response.append([float(j)])
            image_names.append(img_path)

    testData = np.float32(feature_mat).reshape(-1,94)
    responses = np.float32(response)
    result = svm_model.predict_all(testData)
    mask = result==responses

	#calculate calories
    for i in range(0, len(result)):
        volume = (getVolume(result[i], fruit_areas[i], skin_areas[i], pix_cm[i], fruit_contours[i]))
        mass, cal, cal_100 = getCalorie(result[i], volume)
        fruit_volumes.append(volume)
        fruit_calories.append(cal)
        fruit_calories_100grams.append(cal_100)
        fruit_mass.append(mass)

	#write into csv file
    with open('output.csv','w') as outfile:
       
       writer = csv.writer (outfile)
       data = ["Image name", "Desired response", "Output label", "Volume (cm^3)", "Mass (grams)", "Calories for food item", "Calories per 100 grams"]
       writer.writerow(data)
       for i in range(0, len(result)):
           if (fruit_volumes[i] == None):
               data = [str(image_names[i]), str(responses[i][0]), str(result[i][0]), "--", "--", "--", str(fruit_calories_100grams[i])]
           else:
               data = [str(image_names[i]), str(responses[i][0]), str(result[i][0]), str(fruit_volumes[i]), str(fruit_mass[i]), str(fruit_calories[i]), str(fruit_calories_100grams[i])]
           writer.writerow(data)
           file.close()
	
    for i in range(0, len(mask)):
        if mask[i][0] == False:	
            print ("Actual Reponse"), responses[i][0], "(Output)", result[i][0], image_names[i]

    correct = np.count_nonzero(mask)
    print (correct*100.0/result.size)
    
if __name__ == '__main__':
    training()
    testing()

