import os

import numpy as np
import pandas as pd
import torch
from PIL import Image
from facenet_pytorch import InceptionResnetV1
from facenet_pytorch.models.mtcnn import MTCNN
from sklearn.metrics.pairwise import cosine_similarity
from torch.utils.data import DataLoader

from CustomDataset import CustomDataset


def collate_fn(batch):
    return batch[0]


class RetrievalComponent:

    def __init__(self):
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        print('Running on device: {}'.format(self.device))

    def calculate_embeddings(self, dataset_folder, embedding_file, batch_size=200):
        workers = 0 if os.name == 'nt' else 2

        # MTCNN (Multi-task Cascaded Convolutional Networks) -> face detection
        mtcnn = MTCNN(
            image_size=160, margin=0, min_face_size=20,
            thresholds=[0.6, 0.7, 0.7], factor=0.709, post_process=True,
            device=self.device
        )

        # InceptionResnetV1 -> face recognition
        resnet = InceptionResnetV1(pretrained='vggface2').eval().to(self.device)

        dataset = CustomDataset(dataset_folder)
        loader = DataLoader(dataset, collate_fn=collate_fn, num_workers=workers)

        print("Starting face detection.")

        aligned = []
        target_variables = []
        total_requests = len(loader)
        elaborated_requests = 0
        for path, x, y in loader:
            try:
                x_aligned, prob = mtcnn(x, return_prob=True)
            except Exception:
                print(x)
                print(path)
                raise
            if x_aligned is not None:
                aligned.append(x_aligned)
                target_variables.append(y)
            elaborated_requests += 1
            if elaborated_requests % batch_size == 0 or elaborated_requests == total_requests:
                print(f"Detected {elaborated_requests} faces out of {total_requests} total faces.")

        print("Face detection ended.")

        print("Start embeddings' calculation.")

        batched_embeddings = []
        for i in range(0, len(aligned), batch_size):
            print('Calulated embedding for {} faces'.format(i))
            batch_aligned = aligned[i:i + batch_size]
            batch_aligned = torch.stack(batch_aligned).to(self.device)
            batch_embeddings = resnet(batch_aligned).detach().cpu().numpy()
            batched_embeddings.append(batch_embeddings)

        embeddings = np.concatenate(batched_embeddings, axis=0)

        print("Embeddings' calculation ended.")

        embeddings_df = pd.DataFrame(embeddings,
                                     columns=["embedding_dim_" + str(i) for i in range(embeddings.shape[1])])
        embeddings_df['gender'] = [t[0] for t in target_variables]
        embeddings_df['age'] = [t[1] for t in target_variables]

        embeddings_df.to_csv(embedding_file, index=False)

    def get_most_similar_embedding(self, embedding_file, image_path, target_variable):

        embeddings_df = pd.read_csv(embedding_file)

        embeddings = embeddings_df.iloc[:, :-2].values

        df_target_variables = embeddings_df[target_variable].values

        resnet = InceptionResnetV1(pretrained='vggface2').eval().to(self.device)

        mtcnn = MTCNN(
            image_size=160, margin=0, min_face_size=20,
            thresholds=[0.6, 0.7, 0.7], factor=0.709, post_process=True,
            device=self.device
        )

        new_image = Image.open(image_path)
        new_image_tensor = mtcnn(new_image)
        new_image_tensor = new_image_tensor.unsqueeze(0)

        new_embedding = resnet(new_image_tensor.to(self.device)).detach().cpu().numpy()

        similarities = cosine_similarity(new_embedding.reshape(1, -1), embeddings)

        maximum_similarity = np.max(similarities)
        if maximum_similarity >= 0.999:
            # Get the index of the second maximum value
            most_similar_index = np.argpartition(similarities[0], -2)[-2]
        else:
            # Find the index of the most similar embedding
            most_similar_index = np.argmax(similarities)

        max_similarity = similarities[0][most_similar_index]

        print("Maximum similarity value:", max_similarity)
        print("Index of the most similar value:", most_similar_index)

        most_similar_target_variable = df_target_variables[most_similar_index]

        print('Most similar {}: {}'.format(target_variable, most_similar_target_variable))

        return most_similar_target_variable, max_similarity, new_embedding
