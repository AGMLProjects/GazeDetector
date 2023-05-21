import os, shutil

if __name__ == "__main__":
    DATASET_PATH = "/home/lorenzo/Gender/"

    # Create the two class directories if not already present
    if os.path.isdir(DATASET_PATH + "male") == False:
        os.makedirs(DATASET_PATH + "male")

    if os.path.isdir(DATASET_PATH + "female") == False:
        os.makedirs(DATASET_PATH + "female")

    # Move the images to the correct class directory
    for filename in os.listdir(DATASET_PATH):
        if filename.endswith(".jpg") == True:
            
            gender = filename.split("_")[1]

            if gender == "0":
                shutil.move(DATASET_PATH + filename, DATASET_PATH + "male/" + filename)
            elif gender == "1":
                shutil.move(DATASET_PATH + filename, DATASET_PATH + "female/" + filename)
