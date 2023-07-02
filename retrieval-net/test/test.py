from mlsocket import MLSocket
import numpy as np
from PIL import Image

HOST = '127.0.0.1'
PORT = 65432

image = Image.open("../data/small_dataset/43_0_2_20170117140033069.jpg")
data = np.array(image)

with MLSocket() as s:
    s.connect((HOST, PORT))
    s.send(data)
    gender = int(s.recv(1024))
    age = int(s.recv(1024))
    similarity = float(s.recv(1024))
    embedding = s.recv(1024)
    print(f"Most similar gender is {gender}, age is {age}, similarity is {similarity}, most similar "
          f"embedding is {embedding}.")
