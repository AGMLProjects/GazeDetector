import os, shutil

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
            