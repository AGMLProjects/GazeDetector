import os

import numpy as np
import pandas as pd
import torch
from facenet_pytorch import InceptionResnetV1
from facenet_pytorch.models.mtcnn import MTCNN
from sklearn.metrics import precision_score, recall_score
from sklearn.metrics.pairwise import cosine_similarity
from torch.utils.data import DataLoader

from CustomDataset import CustomDataset


def collate_fn(batch):
    return batch


class RetrievalComponent:

    def __init__(self):
        self.embeddings_df = None
        np.warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        print('Running on device: {}'.format(self.device))
        self.workers = 0 if os.name == 'nt' else 2
        print('Workers: {}'.format(self.workers))

    def evaluate(self, dataset_folder, train_percentage, batch_size, embedding_file):
        print('Starting evaluation')

        dataset = CustomDataset(dataset_folder)
        train_size = int(train_percentage * len(dataset))
        test_size = len(dataset) - train_size
        train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
        loader = DataLoader(train_dataset, collate_fn=collate_fn, num_workers=self.workers, batch_size=batch_size)

        aligned, target_variables = self.get_face_detection_aligned_and_target(train_size, loader, batch_size)

        self.perform_face_recognition(aligned, target_variables, batch_size, embedding_file)

        ground_truth_gender = []
        ground_truth_age = []
        predicted_gender = []
        predicted_age = []

        evaluated = 0
        for image_path, image, targets in test_dataset:
            try:
                gender, age, similarity, _ = self.get_most_similar_embedding(embedding_file, image)
            except Exception:
                continue

            ground_truth_gender.append(targets[0])
            ground_truth_age.append(targets[1])
            predicted_gender.append(gender)
            predicted_age.append(age)

            evaluated += 1
            if evaluated % batch_size == 0:
                print(f'Evaluated: {evaluated} examples')

        precision_gender = precision_score(ground_truth_gender, predicted_gender, average='micro')
        recall_gender = recall_score(ground_truth_gender, predicted_gender, average='micro')
        precision_age = precision_score(ground_truth_age, predicted_age, average='micro')
        recall_age = recall_score(ground_truth_age, predicted_age, average='micro')
        mae = np.mean(np.abs(np.array(ground_truth_age) - np.array(predicted_age)))

        print(f'Evaluated: {evaluated} examples')
        print(f"Gender precision: {precision_gender}, recall: {recall_gender}.")
        print(f"Age precision: {precision_age}, recall: {recall_age}, mean absolute error: {mae}")
        return precision_gender, recall_gender, precision_age, recall_age, mae

    def perform_face_detection(self, dataset_folder, batch_size):
        print("Starting face detection.")

        dataset = CustomDataset(dataset_folder)
        loader = DataLoader(dataset, collate_fn=collate_fn, num_workers=self.workers, batch_size=batch_size)

        aligned, target_variables = self.get_face_detection_aligned_and_target(len(dataset.image_paths), loader,
                                                                               batch_size)

        print("Face detection ended.")
        return aligned, target_variables

    def get_face_detection_aligned_and_target(self, total_number_of_images, loader, batch_size):
        # MTCNN (Multi-task Cascaded Convolutional Networks) -> face detection
        mtcnn = MTCNN(
            image_size=160, margin=0, min_face_size=20,
            thresholds=[0.6, 0.7, 0.7], factor=0.709, post_process=True,
            device=self.device
        )

        aligned = []
        target_variables = []
        total_batches = len(loader)
        print("Total number of images: {}, using batch size: {}, total number of batches: {}".format(
            total_number_of_images, batch_size, total_batches))
        elaborated_requests = 0
        for batch in loader:
            paths, xs, ys = zip(*batch)

            xs_aligned, probs = mtcnn(xs, return_prob=True)

            for x_aligned, y in zip(xs_aligned, ys):
                if x_aligned is not None:
                    aligned.append(x_aligned)
                    target_variables.append(y)
                elaborated_requests += 1
                if elaborated_requests % batch_size == 0 or elaborated_requests == total_number_of_images:
                    print(f"Detected {elaborated_requests} faces out of {total_number_of_images} total faces.")
        return aligned, target_variables

    def perform_face_recognition(self, aligned, target_variables, batch_size, embedding_file):
        print("Starting embeddings calculation.")

        # InceptionResnetV1 -> face recognition
        resnet = InceptionResnetV1(pretrained='vggface2').eval().to(self.device)

        batched_embeddings = []
        for i in range(0, len(aligned), batch_size):
            print('Calculated embedding for {} faces'.format(i))
            batch_aligned = aligned[i:i + batch_size]
            batch_aligned = torch.stack(batch_aligned).to(self.device)
            batch_embeddings = resnet(batch_aligned).detach().cpu().numpy()
            batched_embeddings.append(batch_embeddings)
        print('Calculated embedding for {} faces'.format(len(aligned)))

        embeddings = np.concatenate(batched_embeddings, axis=0)

        print("Embeddings calculation ended. Saving results...")

        embeddings_df = pd.DataFrame(embeddings,
                                     columns=["embedding_dim_" + str(i) for i in range(embeddings.shape[1])])
        embeddings_df['gender'] = [t[0] for t in target_variables]
        embeddings_df['age'] = [t[1] for t in target_variables]

        embeddings_df.to_csv(embedding_file, index=False)

        print("Embeddings saved to {}".format(embedding_file))

    def calculate_embeddings(self, dataset_folder, embedding_file, batch_size=200):
        aligned, target_variables = self.perform_face_detection(dataset_folder, batch_size)

        self.perform_face_recognition(aligned, target_variables, batch_size, embedding_file)

    def get_most_similar_embedding(self, embedding_file, image):

        embeddings_df = self.load_embeddings(embedding_file)

        embeddings = embeddings_df.iloc[:, :-2].values

        df_gender = embeddings_df['gender'].values
        df_age = embeddings_df['age'].values

        resnet = InceptionResnetV1(pretrained='vggface2').eval().to(self.device)

        mtcnn = MTCNN(
            image_size=160, margin=0, min_face_size=20,
            thresholds=[0.6, 0.7, 0.7], factor=0.709, post_process=True,
            device=self.device
        )

        new_image_tensor = mtcnn(image)
        new_image_tensor = new_image_tensor.unsqueeze(0)

        new_embedding = resnet(new_image_tensor.to(self.device)).detach().cpu().numpy()

        similarities = cosine_similarity(new_embedding.reshape(1, -1), embeddings)

        most_similar_index = np.argpartition(similarities[0], -2)[-2]

        max_similarity = similarities[0][most_similar_index]

        most_similar_gender = df_gender[most_similar_index]
        most_similar_age = df_age[most_similar_index]

        return most_similar_gender, most_similar_age, max_similarity, new_embedding

    def load_embeddings(self, embedding_file):
        if self.embeddings_df is None:
            self.embeddings_df = pd.read_csv(embedding_file)
        return self.embeddings_df
