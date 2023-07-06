from mlsocket import MLSocket
import numpy as np
from PIL import Image
import json

HOST = '127.0.0.1'
PORT = 65432

image = Image.open("../data/small_dataset/43_0_2_20170117140033069.jpg")
data = np.array(image)

with MLSocket() as s:
    s.connect((HOST, PORT))
    s.send(data)
    return_dict = json.loads(s.recv(1024).decode('UTF-8'))
    embedding = s.recv(1024)
    print(f"Most similar gender is {return_dict['gender']}, age is {return_dict['age']}, similarity is {return_dict['similarity']}, "
          f"most similar embedding is {embedding}.")
