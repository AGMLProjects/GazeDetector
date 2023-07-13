from RetrievalComponent import RetrievalComponent

retrieval_component = RetrievalComponent()
retrieval_component.evaluate("../../Resources/Gender", 0.85, 200, "../../Resources/RetrievalEmbeddings/eval_embeddings.csv")
