from RetrievalComponent import RetrievalComponent
from mlsocket import MLSocket

HOST = '127.0.0.1'
PORT = 65432

if __name__ == '__main__':
    retrievalComponent = RetrievalComponent()

    # uncomment this line if you want to also re-calculate embeddings
    # retrievalComponent.calculate_embeddings('data/small_dataset', "data/output/embeddings.csv")

    with MLSocket() as s:
        s.bind((HOST, PORT))
        s.listen()

        print('Waiting for requests...')

        while True:
            conn, address = s.accept()

            with conn:
                data = conn.recv(1024)
                print('Received: {}'.format(data))
                gender, age, similarity, embedding = retrievalComponent.get_most_similar_embedding(
                    "data/output/embeddings.csv", data)

                print(f"Most similar gender is {gender}, age is {age}, similarity is {similarity}, most similar "
                      f"embedding is {embedding}.")
                conn.send(str(int(gender)).encode('utf-8'))
                conn.send(str(int(age)).encode('utf-8'))
                conn.send(str(float(similarity)).encode('utf-8'))
                conn.send(embedding)
