import pandas as pd
import numpy as np
from PIL import Image
from facenet_pytorch import InceptionResnetV1
import torch
from facenet_pytorch.models.mtcnn import MTCNN
from sklearn.metrics.pairwise import cosine_similarity

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Load the embeddings from the CSV file
embeddings_df = pd.read_csv("data/output/embeddings.csv")

# Convert embeddings to a NumPy array
embeddings = embeddings_df.iloc[:, :-1].values

# Get the corresponding genders
genders = embeddings_df["gender"].values

# Load the face recognition module (Inception Resnet V1)
resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)

# Load and preprocess the new image
mtcnn = MTCNN(
    image_size=160, margin=0, min_face_size=20,
    thresholds=[0.6, 0.7, 0.7], factor=0.709, post_process=True,
    device=device
)

new_image = Image.open("data/small_organized_dataset/0/43_0_2_20170117140033069.jpg")
new_image_tensor = mtcnn(new_image)
new_image_tensor = new_image_tensor.unsqueeze(0)

# Calculate the embedding for the new image
new_embedding = resnet(new_image_tensor.to(device)).detach().cpu().numpy()

# Calculate the cosine similarity between the new embedding and all the previous embeddings
similarities = cosine_similarity(new_embedding.reshape(1, -1), embeddings)

# Find the index of the most similar embedding
most_similar_index = np.argmax(similarities)

# Get the corresponding gender
most_similar_gender = genders[most_similar_index]

print("Most similar gender: ", most_similar_gender)
