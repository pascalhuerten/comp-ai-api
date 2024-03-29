from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field, validator
from typing import List, Literal
from ..models.ComplevelPredictor import ComplevelPredictor
from ..models.ComplevelModelTrainer import ComplevelModelTrainer

router = APIRouter()


class PredictCompLevelRequest(BaseModel):
    title: str = Field(default="")
    description: str = Field(default="")

class TextLabelItem(BaseModel):
    text: str = Field(...)
    label: str = Field(...)

class TrainCompLevelRequest(BaseModel):
    data: List[TextLabelItem] = Field(...)

class CompLevelResponseV1(BaseModel):
    class_probability: List[float] = Field(...)
    level: Literal["A", "B", "C"] = Field(
        ..., description="The predicted level. Will be one of 'A', 'B', or 'C'. Note: If the model predicts 'D', it will be converted to 'C'."
    )
    target_probability: float = Field(...)

    @validator("level")
    def check_level(cls, v):
        if v not in ["A", "B", "C"]:
            raise ValueError('level must be one of "A", "B", "C"')
        return v

@router.post("/predictCompLevel", response_model=CompLevelResponseV1)
def predict_complevel(request: PredictCompLevelRequest):
    model = ComplevelPredictor()
    prediction = model.predict(request.title, request.description)
    
    # Check if the level prediction is 'D'
    if prediction["level"] == 'D':
        # If it is, change it to 'C'
        prediction["level"] = 'C'
    
    return CompLevelResponseV1(class_probability=prediction["class_probability"], level=prediction["level"], target_probability=prediction["target_probability"])

class CompLevelResponseV2(BaseModel):
    class_probability: List[float] = Field(...)
    level: Literal["A", "B", "C", "D"] = Field(
        ..., description="The predicted level. Will be one of 'A', 'B', 'C', or 'D'. Unlike version 1, 'D' is a possible output in this version."
    )
    target_probability: float = Field(...)

    @validator("level")
    def check_level(cls, v):
        if v not in ["A", "B", "C", "D"]:
            raise ValueError('level must be one of "A", "B", "C", or "D"')
        return v

@router.post("/v2/predictCompLevel", response_model=CompLevelResponseV2)
def predict_complevel(request: PredictCompLevelRequest):
    model = ComplevelPredictor()
    prediction = model.predict(request.title, request.description)
    return CompLevelResponseV2(class_probability=prediction["class_probability"], level=prediction["level"], target_probability=prediction["target_probability"])

@router.post("/trainCompLevel")
def train_complevel(request: TrainCompLevelRequest):
    trainer = ComplevelModelTrainer()
    training_stats = trainer.train(request.data)
    return training_stats


@router.get("/getCompLevelReport")
def report_complevel():
    trainer = ComplevelModelTrainer()
    report = trainer.getReport()
    if report is None:
        raise HTTPException(status_code=404, detail="Report not found")
    return report
