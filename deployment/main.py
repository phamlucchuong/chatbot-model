from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Dict, Any
from predictor import SymptomPredictor
import uvicorn
from datetime import datetime


# Initialize FastAPI
app = FastAPI(
    title="Healthcare Chatbot API",
    description="API nhận dạng triệu chứng và dự đoán bệnh sử dụng PhoBERT NER và Naive Bayes",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize predictor
predictor = SymptomPredictor(models_dir="../training/models")


# ==================== REQUEST/RESPONSE MODELS ====================

class SymptomExtractionRequest(BaseModel):
    """Request để trích xuất triệu chứng"""
    content: str = Field(..., description="Văn bản mô tả triệu chứng", min_length=1)
    
    class Config:
        json_schema_extra = {
            "example": {
                "content": "Tôi bị sốt cao, đau đầu dữ dội và ho nhiều"
            }
        }


class SymptomExtractionResponse(BaseModel):
    """Response trả về danh sách triệu chứng"""
    symptoms: List[str] = Field(..., description="Danh sách triệu chứng được nhận dạng")
    # count: int = Field(..., description="Số lượng triệu chứng")
    
    class Config:
        json_schema_extra = {
            "example": {
                "symptoms": ["sốt cao", "đau đầu", "ho"],
                # "count": 3
            }
        }


class DiseasePredictionRequest(BaseModel):
    """Request để dự đoán bệnh"""
    symptoms: List[str] = Field(..., description="Danh sách triệu chứng", min_items=1)
    top_k: int = Field(5, description="Số lượng dự đoán hàng đầu", ge=1, le=10)
    
    class Config:
        json_schema_extra = {
            "example": {
                "symptoms": ["sốt cao", "đau đầu", "ho", "đau cơ"],
                "top_k": 5
            }
        }


class TopPrediction(BaseModel):
    """Một dự đoán trong top K"""
    disease_id: str = Field(..., description="Mã bệnh (vd: D001)")
    disease_name: str = Field(..., description="Tên bệnh")
    confidence: float = Field(..., description="Độ tin cậy (0-1)", ge=0, le=1)


class DiseasePredictionResponse(BaseModel):
    """Response trả về kết quả dự đoán"""
    disease_id: str = Field(..., description="Mã bệnh được dự đoán")
    disease_name: str = Field(..., description="Tên bệnh được dự đoán")
    confidence: float = Field(..., description="Độ tin cậy", ge=0, le=1)
    matched_symptoms: List[str] = Field(..., description="Triệu chứng khớp với model")
    unmatched_symptoms: List[str] = Field(..., description="Triệu chứng không khớp")
    top_predictions: List[TopPrediction] = Field(..., description="Top K dự đoán")
    
    class Config:
        json_schema_extra = {
            "example": {
                "disease_id": "D002",
                "disease_name": "Cảm cúm (Influenza)",
                "confidence": 0.85,
                "matched_symptoms": ["sốt cao", "đau đầu", "ho"],
                "unmatched_symptoms": [],
                "top_predictions": [
                    {"disease_id": "D002", "disease_name": "Cảm cúm (Influenza)", "confidence": 0.85},
                    {"disease_id": "D001", "disease_name": "Cảm lạnh", "confidence": 0.10}
                ]
            }
        }


class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    message: str
    timestamp: str
    models_loaded: bool
    device: str


# ==================== STARTUP EVENT ====================

@app.on_event("startup")
async def startup_event():
    """Load models khi khởi động server"""
    print("\n" + "="*70)
    print("🚀 STARTING HEALTHCARE CHATBOT API")
    print("="*70)
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Author: phamlucchuong")
    print("="*70)
    
    try:
        predictor.load_models()
        print("\n✅ Server started successfully!")
        print(f"📍 Docs: http://localhost:8000/docs")
        print(f"📍 API: http://localhost:8000/api/")
        print("="*70 + "\n")
    except Exception as e:
        print(f"\n❌ Failed to start server: {str(e)}")
        raise e


# ==================== ENDPOINTS ====================

@app.get("/", response_model=HealthResponse)
async def root():
    """Health check endpoint"""
    return HealthResponse(
        status="healthy",
        message="Healthcare Chatbot API is running",
        timestamp=datetime.now().isoformat(),
        models_loaded=predictor.phobert_model is not None and predictor.bayes_model is not None,
        device=predictor.device
    )


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Detailed health check"""
    return HealthResponse(
        status="healthy",
        message="All systems operational",
        timestamp=datetime.now().isoformat(),
        models_loaded=predictor.phobert_model is not None and predictor.bayes_model is not None,
        device=predictor.device
    )


@app.post("/api/extract-symptoms", response_model=SymptomExtractionResponse)
async def extract_symptoms(request: SymptomExtractionRequest):
    """
    Trích xuất triệu chứng từ văn bản sử dụng PhoBERT NER
    
    - **content**: Văn bản mô tả triệu chứng (tiếng Việt)
    
    Returns danh sách các triệu chứng được nhận dạng
    """
    try:
        symptoms = predictor.extract_symptoms(request.content)
        return SymptomExtractionResponse(
            symptoms=symptoms,
            count=len(symptoms)
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Lỗi khi trích xuất triệu chứng: {str(e)}")


@app.post("/api/predict-disease", response_model=DiseasePredictionResponse)
async def predict_disease(request: DiseasePredictionRequest):
    """
    Dự đoán bệnh từ danh sách triệu chứng sử dụng Naive Bayes
    
    - **symptoms**: Danh sách các triệu chứng (ít nhất 1)
    - **top_k**: Số lượng dự đoán hàng đầu (mặc định: 5)
    
    Returns thông tin bệnh được dự đoán, độ tin cậy, và top K predictions
    """
    try:
        if not request.symptoms:
            raise HTTPException(
                status_code=400, 
                detail="Danh sách triệu chứng không được rỗng"
            )
        
        result = predictor.predict_disease(request.symptoms, top_k=request.top_k)
        
        return DiseasePredictionResponse(**result)
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Lỗi khi dự đoán bệnh: {str(e)}")


@app.post("/api/full-prediction")
async def full_prediction(request: SymptomExtractionRequest):
    """
    Pipeline đầy đủ: Trích xuất triệu chứng → Dự đoán bệnh
    
    - **content**: Văn bản mô tả triệu chứng
    
    Returns kết quả đầy đủ bao gồm triệu chứng và dự đoán bệnh
    """
    try:
        # Step 1: Extract symptoms
        symptoms = predictor.extract_symptoms(request.content)
        
        if not symptoms:
            return {
                "message": "Không nhận dạng được triệu chứng nào",
                "symptoms": [],
                "prediction": None
            }
        
        # Step 2: Predict disease
        prediction = predictor.predict_disease(symptoms, top_k=5)
        
        return {
            "message": "Thành công",
            "input_text": request.content,
            "extracted_symptoms": symptoms,
            "prediction": prediction
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Lỗi: {str(e)}")


# ==================== RUN SERVER ====================

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )