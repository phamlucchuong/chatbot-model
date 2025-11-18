import spacy
import pickle
import json
from pathlib import Path
from typing import List, Dict, Any


class SymptomPredictor:
    """Class để xử lý việc load và sử dụng các model"""
    
    def __init__(self, models_dir: str = "models"):
        self.models_dir = Path(models_dir)
        self.ner_model = None
        self.bayes_model = None
        self.model_info = None
        
    def load_models(self):
        """Load tất cả các models cần thiết"""
        try:
            # Load spaCy NER model
            ner_model_path = self.models_dir / "spacy_ner_model"
            print(f"Loading NER model from {ner_model_path}...")
            self.ner_model = spacy.load(ner_model_path)
            print("NER model loaded successfully!")
            
            # Load Naive Bayes model
            bayes_model_path = self.models_dir / "naive_bayes_model.pkl"
            print(f"Loading Bayes model from {bayes_model_path}...")
            with open(bayes_model_path, 'rb') as f:
                self.bayes_model = pickle.load(f)
            print("Bayes model loaded successfully!")
            
            # Load model info (chứa danh sách symptoms và diseases)
            info_path = self.models_dir / "model_info.json"
            print(f"Loading model info from {info_path}...")
            with open(info_path, 'r', encoding='utf-8') as f:
                self.model_info = json.load(f)
            print("Model info loaded successfully!")
            
            return True
        except Exception as e:
            print(f"Error loading models: {str(e)}")
            raise e
    
    def extract_symptoms(self, content: str) -> List[str]:
        # ...
        doc = self.ner_model(content)
        
        # (Giữ các dòng print debug của bạn)
        print(f"[DEBUG] VĂN BẢN ĐẦU VÀO: {content}")
        print(f"[DEBUG] MODEL TÌM THẤY (doc.ents): {doc.ents}")
        
        symptoms = []
        for ent in doc.ents:
            print(f"[DEBUG] ĐÃ TÌM THẤY ENTITY: '{ent.text}', VỚI LABEL: '{ent.label_}'")

            # --- SỬA LỖI Ở ĐÂY ---
            # Sửa lại thành "SYMPTOM" (khớp với log)
            if ent.label_ == "SYMPTOM": 
                symptoms.append(ent.text)
            # --- KẾT THÚC SỬA LỖI ---
        
        print(f"[DEBUG] KẾT QUẢ LỌC CUỐI CÙNG: {symptoms}")
        return symptoms
    
    def predict_disease(self, symptoms: List[str]) -> Dict[str, Any]:
        """
        Sử dụng Naive Bayes model để dự đoán bệnh từ danh sách triệu chứng
        
        Args:
            symptoms: List các tên triệu chứng
            
        Returns:
            Dictionary chứa bệnh dự đoán và xác suất
        """
        if self.bayes_model is None or self.model_info is None:
            raise ValueError("Bayes model hoặc model info chưa được load. Vui lòng gọi load_models() trước.")
        
        # Tạo feature vector từ danh sách triệu chứng
        # Feature vector có độ dài bằng số lượng triệu chứng trong model_info
        all_symptoms = self.model_info['symptoms']
        feature_vector = [0] * len(all_symptoms)
        
        # Đánh dấu các triệu chứng có trong input
        for symptom in symptoms:
            if symptom in all_symptoms:
                idx = all_symptoms.index(symptom)
                feature_vector[idx] = 1
        
        # Dự đoán bệnh
        prediction = self.bayes_model.predict([feature_vector])[0]
        
        # Lấy xác suất cho mỗi class
        probabilities = self.bayes_model.predict_proba([feature_vector])[0]
        max_prob = max(probabilities)
        
        # Lấy tên bệnh - model trả về tên bệnh trực tiếp (string), không phải index
        disease_name = str(prediction)
        
        return {
            "disease": disease_name,
            "confidence": float(max_prob),
            "matched_symptoms": [s for s in symptoms if s in all_symptoms],
            "unmatched_symptoms": [s for s in symptoms if s not in all_symptoms]
        }