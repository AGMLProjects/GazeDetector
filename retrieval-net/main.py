from RetrievalComponent import RetrievalComponent
from mlsocket import MLSocket
import json

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

                try:
                    gender, age, similarity, embedding = retrievalComponent.get_most_similar_embedding(
                        "data/output/embeddings.csv", data)

                    print(f"Most similar gender is {gender}, age is {age}, similarity is {similarity}.")

                    return_dict = {'status': True, 'gender': int(gender), 'age': int(age),
                                   'similarity': float(similarity)}
                    conn.send(json.dumps(return_dict).encode('UTF-8'))
                    conn.send(embedding)
                except Exception as e:
                    print(f'Some exception occurred in finding most similar embedding. Error is: {e}')
                    return_dict = {'status': False}
                    conn.send(json.dumps(return_dict).encode('UTF-8'))


