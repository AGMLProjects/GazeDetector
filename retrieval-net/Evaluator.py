from RetrievalComponent import RetrievalComponent

retrieval_component = RetrievalComponent()
retrieval_component.evaluate("./data/small_dataset", 0.75, 200, "./data/eval_embeddings.csv")
