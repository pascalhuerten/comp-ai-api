import json
from langchain.llms import HuggingFaceTextGenInference
from sklearn.metrics.pairwise import cosine_similarity
import re
from openai import OpenAI


class chatsearcher:
    def __init__(self, embedding, skillfit_model, skilldbs, data):
        self.embedding = embedding
        self.skillfit_model = skillfit_model

        self.skill_taxonomy = data.get("skill_taxonomy", "ESCO")
        if self.skill_taxonomy not in skilldbs:
            raise ValueError("Invalid skill_taxonomy value.")
        else:
            self.skilldb = skilldbs[self.skill_taxonomy]

        self.doc = data.get("doc")
        self.los = data.get("los", [])
        self.validated_skills = data.get("skills", [])
        self.validated_skill_uris = [skill["uri"] for skill in self.validated_skills]
        self.valid_skill_labels = [
            skill["title"] for skill in self.validated_skills if skill["valid"]
        ]
        self.filterconcepts = data.get("filterconcepts", [])
        self.top_k = int(data.get("top_k", 20))
        self.strict = int(data.get("strict", 0))
        self.trusted_score = float(data.get("trusted_score", 0.2))
        self.temperature = float(data.get("temperature", 0.05))
        self.use_llm = bool(data.get("use_llm", False))
        self.llm_validation = bool(data.get("llm_validation", False))
        self.skillfit_validation = bool(data.get("skillfit_validation", False))
        self.openai_api_key = data.get("openai_api_key", None)
        self.score_cutoff = data.get("score_cutoff", 0)

    def predict(self):
        if len(self.los) > 0:
            learningoutcomes = "\n".join(self.los)
            searchterms = self.los
        elif self.use_llm:
            learningoutcomes = self.extract_learning_outcomes(self.doc)
            searchterms = learningoutcomes.split("\n")
        else:
            learningoutcomes = self.doc
            searchterms = []

        learningoutcomes = "\n".join(self.valid_skill_labels) + "\n" + learningoutcomes
        searchterms.extend(self.valid_skill_labels)
        embedded_doc = self.embedding.embed_documents([learningoutcomes])

        # Do Vector Search to find most similar skills.
        similar_skills = self.skilldb.similarity_search_with_score(
            learningoutcomes, min(self.top_k * 2, 50) + len(self.validated_skills)
        )

        predictions = [
            self.create_prediction(skill[0], skill[1]) for skill in similar_skills
        ]

        # Filter out skills that are already known or not part of the filterconcepts.
        predictions = self.filter_predictions(predictions)

        # Define artificial threshholds for relevancy by identifying where the similarity rating decreases the fastest.
        if not self.llm_validation and not self.skillfit_validation:
            predictions = self.applyDynamicThreshold(predictions)

        # Predictions based on the known skills.
        predictions = self.finetune_on_validated_skills(predictions)

        # Reduce amount of predictions before performance hungry validation.
        predictions = predictions[: int(self.top_k * 1.3)]

        # Validate predictions.
        if self.llm_validation:
            predictions = self.validate_with_llm(predictions)
        elif self.skillfit_validation and self.skill_taxonomy == "ESCO":
            predictions = self.validate_with_skillfit(predictions, embedded_doc)

        # Sort predictions by score.
        predictions = sorted(predictions, key=lambda x: x["score"], reverse=False)

        # Remove predictions with a score higher than the score_cutoff.
        if self.score_cutoff > 0 and self.score_cutoff < 1:
            predictions = [
                prediction
                for prediction in predictions
                if prediction["score"] < self.score_cutoff
            ]

        # Clean up predictions.
        for prediction in predictions:
            if "broaderConcepts" in prediction:
                del prediction["broaderConcepts"]
            if "penalty" in prediction:
                del prediction["penalty"]
            if (
                "fit" in prediction
                and not self.llm_validation
                and not self.skillfit_validation
            ):
                del prediction["fit"]

        # Return predictions.
        return searchterms, predictions[: self.top_k]

    def apply_llm_validation(self, predictions, errors):
        validated = []
        for i in range(len(predictions)):
            fit = predictions[i]["title"] not in errors
            predictions[i]["fit"] = fit
            if fit:
                if self.strict > 0:
                    penalty = -0.06
                    predictions[i]["penalty"] += penalty
                    predictions[i]["score"] += penalty
                if self.strict > 1:
                    validated.append(predictions[i])
        return validated if (self.strict > 1 and validated) else predictions

    def validate_with_llm(self, predictions):
        skilllabels = [prediction["title"] for prediction in predictions]

        system = "Es soll geprüft werden welche Lernziele von dem folgenden Kurs vermittelt werden. \nKursbeschreibung: "
        user = "Kursbeschreibung: \n" + self.doc[:3500]
        user += (
            "\n\n" + "Welche der folgenden Kompetenzen werden nicht im Kurs vermittelt?"
        )
        user += "\n\n".join(skilllabels)
        user += "\n\nDie Kompetenzen, die nicht zur Kursbeschreibung passen, sind:"
        if self.openai_api_key:
            client = OpenAI()

            completion = client.chat.completions.create(
                model="gpt-3.5-turbo-1106",
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                seed=42,
            )

            chatresponse = completion.choices[0].message.content
        else:
            prompt = system + "\n\n" + user
            llm = HuggingFaceTextGenInference(
                inference_server_url="https://em-german-70b.llm.mylab.th-luebeck.dev/",
                max_new_tokens=512,
                temperature=self.temperature,
            )
            chatresponse = llm(prompt)

        lines = chatresponse.split("\n")
        lines = [line.strip() for line in lines]
        # strip 1. 2. etc or - or * from start of line
        lines = [
            re.sub(r"^ *[\d.-]+ *|^ *\* *|^ *- *", "", line, flags=re.MULTILINE)
            for line in lines
        ]
        # remove empty lines
        lines = [line for line in lines if line]
        return self.apply_llm_validation(predictions, lines)

    def validate_with_skillfit(self, predictions, embedded_doc):
        skillfit_predictions = self.skillfit_model.predict(
            self.skillfit_model.prepare_data_for_prediction(
                self.embedding, embedded_doc, predictions
            )
        )
        return self.apply_skillfit_validation(predictions, skillfit_predictions)

    def apply_skillfit_validation(self, predictions, skillfit_predictions):
        validated = []
        if len(predictions) != len(skillfit_predictions):
            raise ValueError(
                "Expected same number of predictions and skillfit predictions."
            )

        for i in range(len(predictions)):
            fit = bool(skillfit_predictions[i][0])
            predictions[i]["fit"] = fit
            if fit:
                penalty = -0.05
            else:
                penalty = 0.05
            
            if self.strict > 0:
                predictions[i]["penalty"] += penalty
                predictions[i]["score"] += penalty
            if self.strict > 1 and fit:
                validated.append(predictions[i])
        
        return validated if (self.strict > 1) else predictions

    def filter_predictions(self, predictions):
        seen = set()
        filtered = []
        for prediction in predictions:
            if (
                prediction["uri"] not in seen
                and not self.is_known_skill(prediction)
                and self.is_part_of_concept(prediction)
            ):
                seen.add(prediction["uri"])
                filtered.append(prediction)
        return filtered

    def extract_learning_outcomes(self, doc):
        system = "Du bist ein KI-Assistent, der auf Kursbeschreibungen spezialisiert ist. Deine Aufgabe ist es, eine Liste von Lernzielen aus einer gegebenen Kursbeschreibung zu extrahieren. Benannte Vorraussetzunge und Zielgruppen sollen ignoriert werden. Gehe dabei Schritt für Schritt vor. Erfasse zuerst die groben Themen, dann identifiziere zu jedem Thema die vermittelten Fähigkeiten."
        user = "Bitte identifiziere die Lernziele in der folgenden Kursbeschreibung:"
        user += "\n\n" + doc
        user += "\n\nErstelle eine Liste der Lernziele, wobei jedes Lernziel in einer neuen Zeile stehen soll. Die Antwort sollte nur die Lernziele selbst enthalten, ohne Einleitungen oder zusätzliche Worte. Nutze kurze und einfache Sprache sowie BLOOM-Verben für Fähigkeiten."

        if self.openai_api_key:
            client = OpenAI()

            completion = client.chat.completions.create(
                model="gpt-3.5-turbo-1106",
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                seed=42,
            )

            learningoutcomes = completion.choices[0].message.content
        else:
            prompt = system + "\n\n" + user
            max_tokens = 4000
            max_new_tokens = 512
            max_input_tokens = max_tokens - max_new_tokens
            prompt = prompt[:max_input_tokens]
            llm = HuggingFaceTextGenInference(
                inference_server_url="https://em-german-13b.llm.mylab.th-luebeck.dev",
                max_new_tokens=max_new_tokens,
                temperature=self.temperature,
            )
            learningoutcomes = llm(prompt)

        # Remove String " ASSISTANT: " from start of learningoutcomes
        learningoutcomes = learningoutcomes.replace("ASSISTANT: ", "").strip()
        # Remove list decorations using regular expressions
        learningoutcomes = re.sub(
            r"^ *[\d.-]+ *|^ *\* *|^ *- *", "", learningoutcomes, flags=re.MULTILINE
        )

        return learningoutcomes

    def is_known_skill(self, skill):
        return (
            self.skill_taxonomy == "ESCO" and skill["uri"] in self.validated_skill_uris
        )

    def is_part_of_concept(self, skill):
        if len(self.filterconcepts) == 0 or self.skill_taxonomy != "ESCO":
            return True

        if skill["broaderConcepts"]:
            for broaderconcept in skill["broaderConcepts"]:
                if broaderconcept in self.filterconcepts:
                    return True
        return False

    def create_prediction(self, skill, score):
        if self.skill_taxonomy == "ESCO":
            return {
                "uri": skill.metadata["conceptUri"],
                "title": skill.metadata["preferredLabel"],
                "broaderConcepts": [
                    concept["uri"]
                    for concept in json.loads(
                        skill.metadata["broaderHierarchyConcepts"]
                    )
                ]
                if skill.metadata["broaderHierarchyConcepts"]
                else [],
                "score": score,
                "penalty": 0,
                "fit": True,
            }
        else:
            return {
                "uri": skill.metadata["id"],
                "title": skill.page_content,
                "score": score,
                "penalty": 0,
                "fit": True,
            }

    def applyDynamicThreshold(self, predictions):
        if self.strict == 0 or len(predictions) <= 2:
            return predictions

        # Identify the biggest and second biggest gap between the skills with scores higher than 0.2.
        gaps = []
        for i in range(len(predictions) - 1):
            gaps.append(predictions[i + 1]["score"] - predictions[i]["score"])

        # Get the indices of the two largest gaps.
        max_gap_skill_index = gaps.index(max(gaps)) + 1
        if self.strict == 3:
            predictions = predictions[:max_gap_skill_index]
        elif self.strict <= 2:
            max_gap = 0
            max_gap_skill_index_2 = 0
            for i in range(max_gap_skill_index + 1, len(predictions) - 1):
                if predictions[i]["score"] < self.trusted_score:
                    continue
                gap = predictions[i + 1]["score"] - predictions[i]["score"]
                if gap > max_gap:
                    max_gap = gap
                    max_gap_skill_index_2 = i

            if self.strict == 1:
                max_gap = 0
                max_gap_skill_index_3 = 0
                for i in range(max_gap_skill_index_2 + 1, len(predictions) - 1):
                    if predictions[i]["score"] < self.trusted_score:
                        continue
                    gap = predictions[i + 1]["score"] - predictions[i]["score"]
                    if gap > max_gap:
                        max_gap = gap
                        max_gap_skill_index_3 = i

                predictions = predictions[: max_gap_skill_index_3 + 1]
            else:
                predictions = predictions[: max_gap_skill_index_2 + 1]

        return predictions

    def finetune_on_validated_skills(self, predictions):
        if len(self.validated_skills) == 0:
            return predictions
        # Predictions based on validated skills.
        validSkillUris = [
            skill["uri"] for skill in self.validated_skills if skill["valid"]
        ]
        validSkillLabels = "\n".join(
            [skill["title"] for skill in self.validated_skills if skill["valid"]]
        )

        # Do Vector Search to find most similar skills.
        valid_docs = self.skilldb.similarity_search_with_score(validSkillLabels, 10)
        # Create predictions for similar skills and filter out the current skill.
        similarToValidSkills = [
            self.create_prediction(valid_doc[0], valid_doc[1])
            for valid_doc in valid_docs
            if valid_doc[0] not in validSkillUris
        ]
        similarToValidSkills = self.filter_predictions(similarToValidSkills)

        # Add skills that are similar to valid skills and reward them with a higher score.
        for similarValidSkill in similarToValidSkills:
            found = False
            for prediction in predictions:
                if prediction["uri"] == similarValidSkill["uri"]:
                    penalty = -((1 - similarValidSkill["score"]) ** 4) * 0.3
                    prediction["penalty"] += penalty
                    prediction["score"] += penalty
                    found = True
                    break
            if not found:
                predictions.append(similarValidSkill)

        invalidSkillUris = [
            skill["uri"] for skill in self.validated_skills if not skill["valid"]
        ]
        invalidSkillLabels = "\n".join(
            [skill["title"] for skill in self.validated_skills if not skill["valid"]]
        )
        # Do Vector Search to find most similar skills.
        invalid_docs = self.skilldb.similarity_search_with_score(invalidSkillLabels, 10)
        # Create predictions for similar skills and filter out the current skill.
        similarToInvalidSkills = [
            self.create_prediction(invalid_doc[0], invalid_doc[1])
            for invalid_doc in invalid_docs
            if invalid_doc[0] not in invalidSkillUris
        ]
        similarToInvalidSkills = self.filter_predictions(similarToInvalidSkills)
        # Penalty for predictions that are similar to invalid skills.
        for similarInvalidSkill in similarToInvalidSkills:
            for prediction in predictions:
                if prediction["uri"] == similarInvalidSkill["uri"]:
                    # The lower the score, the higher the penalty.
                    penalty = ((1 - similarValidSkill["score"]) ** 4) * 0.5
                    prediction["penalty"] += penalty
                    prediction["score"] += penalty
                    break

        return self.filter_predictions(predictions)
