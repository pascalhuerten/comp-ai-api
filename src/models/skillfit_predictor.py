import pickle
import os
from tensorflow import keras
import numpy as np


class skillfit_predictor:
    def __init__(self):
        self.optimal_threshold = 0.5
        # Load the saved scaler
        self.scaler = pickle.load(
            open("./data/models/skillfit_ai/skillfit_scaler.pickle", "rb")
        )
        self.model = keras.models.load_model(
            "./data/models/skillfit_ai/skillfit_ai-model"
        )

    def prepare_data_for_prediction(self, embedding, embedded_doc, skill_data):
        # Extract skill embeddings and similarity scores
        skill_embeddings = [
            embedding.embed_documents([item["title"]])[0] for item in skill_data
        ]
        similarity_scores = [item["score"] for item in skill_data]

        # Convert to numpy arrays
        skill_embeddings = np.array(skill_embeddings)
        similarity_scores = np.array(similarity_scores)

        # Reshape similarity_scores to be a column vector
        similarity_scores = similarity_scores.reshape(-1, 1)

        # Repeat the embedded_doc and similarity_scores for each skill_embedding
        num_skills = len(skill_embeddings)
        embedded_doc_repeated = np.repeat(embedded_doc, num_skills, axis=0)

        # Flatten the arrays and concatenate them to form the input data
        prepared_data = np.concatenate(
            (embedded_doc_repeated, skill_embeddings, similarity_scores), axis=1
        )

        return prepared_data

    def predict(self, X):
        # Scale your new data
        X_scaled = self.scaler.transform(X)
        y_pred_probs = self.model.predict(X_scaled)
        return (y_pred_probs >= self.optimal_threshold).astype(int).tolist()
