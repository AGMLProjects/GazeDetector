from RetrievalComponent import RetrievalComponent

if __name__ == '__main__':
    retrievalComponent = RetrievalComponent()

    retrievalComponent.calculate_embeddings('data/small_dataset', "data/output/embeddings.csv")

    value, similarity, embedding = retrievalComponent.get_most_similar_embedding(
        "data/output/embeddings.csv", "data/small_dataset/43_0_2_20170117140033069.jpg", 'age')

    print(f"Most similar target variable is {value}, similarity is {similarity}, most similar embedding is {embedding}.")
