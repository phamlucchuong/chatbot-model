from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
from predictor import SymptomPredictor
import uvicorn


# Khởi tạo FastAPI app
app = FastAPI(
    title="Healthcare Chatbot API",
    description="API nhận dạng triệu chứng và dự đoán bệnh",
    version="1.0.0"
)

# Khởi tạo predictor
predictor = SymptomPredictor(models_dir="models")


# Pydantic models cho request/response
class SymptomExtractionRequest(BaseModel):
    content: str
    
    class Config:
        json_schema_extra = {
            "example": {
                "content": "Tôi bị sốt cao, đau đầu và ho nhiều"
            }
        }


class SymptomExtractionResponse(BaseModel):
    symptoms: List[str]
    
    class Config:
        json_schema_extra = {
            "example": {
                "symptoms": ["sốt_cao", "đau_đầu", "ho"]
            }
        }


class DiseasePredictionRequest(BaseModel):
    symptoms: List[str]
    
    class Config:
        json_schema_extra = {
            "example": {
                "symptoms": ["sốt_cao", "đau_đầu", "ho", "đau_họng"]
            }
        }


class DiseasePredictionResponse(BaseModel):
    disease: str
    confidence: float
    matched_symptoms: List[str]
    unmatched_symptoms: List[str]
    
    class Config:
        json_schema_extra = {
            "example": {
                "disease": "Cúm",
                "confidence": 0.85,
                "matched_symptoms": ["sốt_cao", "đau_đầu", "ho"],
                "unmatched_symptoms": []
            }
        }


@app.on_event("startup")
async def startup_event():
    """Load models khi khởi động server"""
    print("Loading models...")
    try:
        predictor.load_models()
        print("All models loaded successfully!")
    except Exception as e:
        print(f"Error loading models: {str(e)}")
        raise e


@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "message": "Healthcare Chatbot API is running",
        "status": "healthy",
        "endpoints": {
            "extract_symptoms": "/api/extract-symptoms",
            "predict_disease": "/api/predict-disease",
            "docs": "/docs"
        }
    }


@app.post("/api/extract-symptoms", response_model=SymptomExtractionResponse)
async def extract_symptoms(request: SymptomExtractionRequest):
    """
    Endpoint để nhận dạng triệu chứng từ text
    
    Args:
        request: Object chứa text mô tả triệu chứng
        
    Returns:
        List các triệu chứng được nhận dạng
    """
    try:
        symptoms = predictor.extract_symptoms(request.content)
        return SymptomExtractionResponse(symptoms=symptoms)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error extracting symptoms: {str(e)}")


@app.post("/api/predict-disease", response_model=DiseasePredictionResponse)
async def predict_disease(request: DiseasePredictionRequest):
    """
    Endpoint để dự đoán bệnh từ danh sách triệu chứng
    
    Args:
        request: Object chứa list các triệu chứng
        
    Returns:
        Tên bệnh dự đoán, độ tin cậy và thông tin về triệu chứng
    """
    try:
        if not request.symptoms:
            raise HTTPException(status_code=400, detail="Danh sách triệu chứng không được để trống")
        
        result = predictor.predict_disease(request.symptoms)
        return DiseasePredictionResponse(**result)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error predicting disease: {str(e)}")


if __name__ == "__main__":
    # Chạy server
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )