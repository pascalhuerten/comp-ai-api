# Setup and imports
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)
from scikeras.wrappers import KerasClassifier
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from tensorflow.keras.callbacks import EarlyStopping
from datetime import datetime
import time
import pickle
import os
import json
from flask import current_app


class skillfit_model_trainer:
    def __init__(self):
        self.cur = current_app.config["storage"].cursor
        pass

    def getReport(self):
        if os.path.exists("./data/logs/skillfitReport.json"):
            fileObject = open("./data/logs/skillfitReport.json", "r")
            jsonContent = fileObject.read()
            return json.loads(jsonContent)
        else:
            return None

    def addTrainingData(self, doc, validationResults):
        # insert course
        # Generate hash of course
        doc_id = str(hash(doc))
        self.cur.execute("SELECT id FROM courses WHERE id = %s", (doc_id,))
        if self.cur.fetchone() is None:
            self.cur.execute(
                "INSERT INTO courses (id, text) VALUES (%s, %s)", (doc_id, doc)
            )

        # Delete old relations
        self.cur.execute(
            "DELETE FROM course_skills WHERE course_id = %s", (doc_id,)
        )

        # insert skills
        for skill in validationResults:
            skill_id = skill["uri"]
            self.cur.execute("SELECT id FROM skills WHERE id = %s", (skill_id,))
            if self.cur.fetchone() is None:
                self.cur.execute(
                    "INSERT INTO skills (id, name, taxonomy) VALUES (%s, %s, %s)",
                    (skill_id, skill["title"], skill["taxonomy"]),
                )

            # insert relation
            self.cur.execute(
                "SELECT id FROM course_skills WHERE course_id = %s AND skill_id = %s",
                (doc_id, skill_id),
            )
            if self.cur.fetchone() is None:
                self.cur.execute(
                    "INSERT INTO course_skills (course_id, skill_id, valid) VALUES (%s, %s, %s)",
                    (doc_id, skill_id, skill["valid"]),
                )
            # Update relation
            else:
                self.cur.execute(
                    "UPDATE course_skills SET valid = %s WHERE course_id = %s AND skill_id = %s",
                    (skill["valid"], doc_id, skill_id),
                )

        # Response to client
        return {"status": 200, "message": "Training data added successfully"}

    def getCourseSkills(self):
        self.cur.execute(
            "SELECT d.text, s.id, ds.valid FROM courses d, skills s, course_skills ds WHERE d.id = ds.course_id AND s.id = ds.skill_id"
        )
        return self.cur.fetchall()
        

    def train(self):
        log = "Training results:"
        log += "\nDate: " + datetime.now().strftime("%d/%m/%Y %H:%M:%S")
        # Start time.
        st = time.time()

        # Load dataset
        dataset = pd.read_json(self.dir + "/data/preparation/skillfit_dataset.json")

        report = {}
        report["dataset_size"] = dataset.shape[0]
        log += "\nTraining data size: " + str(dataset.shape[0])

        # Load your dataset (assuming it's in a pandas DataFrame)
        # Replace 'target_column_name' with the actual name of your target column
        target_column_name = "fit"
        y = dataset[target_column_name]

        # Count the occurrences of each class
        class_counts = y.value_counts()

        # Compute class ratios
        total_samples = len(y)
        class_ratios = class_counts / total_samples

        log += "\n\nClass Ratios: " + str(class_ratios.tolist())

        # X is data with the column "fit" removed
        X_doc_embedding = dataset["doc_embedding"]
        X_skill_embedding = dataset["skill_embedding"]
        X_similarity = dataset["similarity"]

        # Convert the embeddings to numpy arrays
        X_doc_embedding = np.array(X_doc_embedding.tolist())
        X_skill_embedding = np.array(X_skill_embedding.tolist())

        # Concatenate the embeddings and similarity into a single feature matrix
        X = np.concatenate(
            (X_doc_embedding, X_skill_embedding, X_similarity.values.reshape(-1, 1)),
            axis=1,
        )

        # Split data into training and test sets
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        # Feature Scaling
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        # Create the directory if it doesn't exist
        if not os.path.exists("./data/models/skillfit_ai"):
            os.makedirs("./data/models/skillfit_ai")

        # Save the scaler to a file
        pickle.dump(
            scaler, open("./data/models/skillfit_ai/skillfit_scaler.pickle", "wb")
        )

        # Handle Class Imbalance using SMOTE
        smote = SMOTE(sampling_strategy="auto", random_state=42)
        X_train, y_train = smote.fit_resample(X_train, y_train)

        # Define a parameter grid to search
        param_grid = {
            "input_dim": [X_train.shape[1]],
            "units": [128],
            "activation": ["relu"],
            "dropout_rate": [0.3],
        }

        # Early Stopping
        early_stopping = EarlyStopping(
            monitor="val_loss", patience=10, restore_best_weights=True
        )

        # Create your model function
        def create_model(input_dim, units, activation, dropout_rate):
            model = Sequential()
            model.add(Dense(units=units, activation=activation, input_dim=input_dim))
            model.add(Dropout(dropout_rate))
            model.add(Dense(units=64, activation="relu"))
            model.add(Dropout(0.2))
            model.add(Dense(units=1, activation="sigmoid"))
            model.compile(
                loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"]
            )
            return model

        # Create the KerasClassifier
        kerasClassifier = KerasClassifier(
            model=create_model,
            epochs=100,
            batch_size=128,
            verbose=0,
            validation_split=0.2,
            callbacks=[early_stopping],
            input_dim=X_train.shape[1],
            units=64,
            activation="relu",
            dropout_rate=0.2,
        )

        # Train the model
        kerasClassifier.fit(X_train, y_train)

        # Save model.
        kerasClassifier.model_.save("./data/models/skillfit_ai/skillfit_ai-model")

        # Get predictions for evaluation dataset.
        y_pred_probs = kerasClassifier.predict(X_test)

        # Initialize variables for optimal threshold and max F1-score
        optimal_threshold = 0
        max_f1_score = 0

        # Initialize binary search range
        low, high = 0, 1

        # Set the number of iterations for the binary search
        num_iterations = 100

        # Perform binary search to find the optimal threshold
        for _ in range(num_iterations):
            threshold = (low + high) / 2
            y_pred = (y_pred_probs >= threshold).astype(int)
            f1 = f1_score(y_test, y_pred)
            if f1 > max_f1_score:
                max_f1_score = f1
                optimal_threshold = threshold
            if f1 < 0.5:
                high = threshold
            else:
                low = threshold

        # Use the optimal threshold for predictions
        y_pred = (y_pred_probs >= optimal_threshold).astype(int)

        log += "\n\nOptimal Threshold for Max F1-Score: " + str(optimal_threshold)
        log += "\n\nMax F1-Score: " + str(max_f1_score)

        # Evaluate the model
        # Accuracy
        accuracy = accuracy_score(y_test, y_pred)
        report["accuracy"] = accuracy
        log += "\n\nTest accuracy: {:.2f}%".format(accuracy * 100)
        # Precision
        precision = precision_score(y_test, y_pred)
        report["precision"] = precision
        log += "\n\nTest precision: {:.2f}%".format(precision * 100)
        # Recall
        recall = recall_score(y_test, y_pred)
        report["recall"] = recall
        log += "\n\nTest recall: {:.2f}%".format(recall * 100)
        # F1 Score
        f1 = f1_score(y_test, y_pred)
        report["f1"] = f1
        log += "\n\nTest f1: {:.2f}%".format(f1 * 100)

        # End time.
        et = time.time()

        # Get Elapsed time.
        elapsed_time = et - st
        log += "\n\nExecution time: " + str(elapsed_time) + " seconds"
        print(log)

        report["time"] = st
        report["executiontime"] = elapsed_time
        report["modelname"] = "skillfit-sequential"

        with open("./data/logs/skillfitReport.json", "w") as jsonFile:
            json.dump(report, jsonFile)

        # Check if "/logs/trainSkillfitModelLog.txt exists.
        if os.path.exists("./data/logs/trainSkillfitModelLog.txt"):
            logFile = open("./data/logs/trainSkillfitModelLog.txt", "a")
        else:
            logFile = open("./data/logs/trainSkillfitModelLog.txt", "w")

        logFile.write(log + "\n\n\n")
        logFile.close()

        return report
