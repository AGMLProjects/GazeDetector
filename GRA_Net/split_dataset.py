from os import listdir
import cv2

<<<<<<< Updated upstream
if __name__ == "__main__":
    DATASET_PATH = "C:\\Users\\ventu\\Desktop\\Nuova cartella"
    # Move the images to the correct class directory
    for filename in os.listdir(DATASET_PATH):
        if filename.endswith(".jpg") == True:
            
            gender = filename.split("_")

            if len(gender) != 4:
                print(filename)
                continue
            
            try:
                if int(gender[1]) not in (0, 1):
                    print(filename)
                    continue
            except:
                print(filename)
            
=======
DIR = "./Gender/"

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
>>>>>>> Stashed changes
