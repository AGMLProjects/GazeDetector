from os import listdir
import cv2


DIR = "./Gender/"
'''CIAOOOO'''

#for filename in listdir('C:/tensorflow/models/research/object_detection/images/train'):
count = 0
for filename in listdir(DIR):
  count += 1
  if filename.endswith(".jpg"):
    print(DIR+filename)
    #cv2.imread('C:/tensorflow/models/research/object_detection/images/train/'+filename)
    cv2.imread(DIR+filename)
      
    if count % 1000.0 == 0:
      n = input("Continuare? ")
      count = 0
