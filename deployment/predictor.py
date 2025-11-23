import pickle
import json
import torch
import numpy as np
from pathlib import Path
from typing import List, Dict, Any
from transformers import AutoTokenizer, AutoModelForTokenClassification


class SymptomPredictor:
    """Class để xử lý việc load và sử dụng các model"""
    
    def __init__(self, models_dir: str = "models"):
        self.models_dir = Path(models_dir)
        self.phobert_tokenizer = None
        self.phobert_model = None
        self.bayes_model = None
        self.model_info = None
        self.disease_mapping = None
        self.label_mapping = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
    def load_models(self):
        """Load tất cả các models cần thiết"""
        try:
            print("="*70)
            print("🚀 LOADING MODELS")
            print("="*70)
            
            # 1. Load PhoBERT NER model
            phobert_model_path = self.models_dir / "phobert_ner_model"
            if not phobert_model_path.exists():
                raise FileNotFoundError(f"PhoBERT model not found at {phobert_model_path}")
            
            print(f"\n📥 Loading PhoBERT NER model from {phobert_model_path}...")
            self.phobert_tokenizer = AutoTokenizer.from_pretrained(str(phobert_model_path))
            self.phobert_model = AutoModelForTokenClassification.from_pretrained(str(phobert_model_path))
            self.phobert_model.to(self.device)
            self.phobert_model.eval()
            print(f"✅ PhoBERT loaded on {self.device}")
            
            # 2. Load PhoBERT label mapping
            label_mapping_path = phobert_model_path / "label_mapping.json"
            with open(label_mapping_path, 'r', encoding='utf-8') as f:
                self.label_mapping = json.load(f)
                self.id2label = {int(k): v for k, v in self.label_mapping['id2label'].items()}
            print(f"✅ Label mapping loaded: {len(self.id2label)} labels")
            
            # 3. Load Naive Bayes model
            bayes_model_path = self.models_dir / "naive_bayes_model.pkl"
            if not bayes_model_path.exists():
                raise FileNotFoundError(f"Bayes model not found at {bayes_model_path}")
            
            print(f"\n📥 Loading Naive Bayes model from {bayes_model_path}...")
            with open(bayes_model_path, 'rb') as f:
                self.bayes_model = pickle.load(f)
            print(f"✅ Naive Bayes loaded: {type(self.bayes_model).__name__}")
            
            # 4. Load model info
            info_path = self.models_dir / "model_info.json"
            if not info_path.exists():
                raise FileNotFoundError(f"Model info not found at {info_path}")
            
            print(f"\n📥 Loading model info from {info_path}...")
            with open(info_path, 'r', encoding='utf-8') as f:
                self.model_info = json.load(f)
            
            print(f"✅ Model info loaded:")
            print(f"   - Symptoms: {len(self.model_info.get('symptoms', []))}")
            print(f"   - Diseases: {len(self.model_info.get('diseases', []))}")
            
            # 5. Load disease mapping
            self._load_disease_mapping()
            
            print("\n" + "="*70)
            print("✅ ALL MODELS LOADED SUCCESSFULLY")
            print("="*70)
            
            return True
            
        except Exception as e:
            print(f"\n❌ Error loading models: {str(e)}")
            raise e
    
    def _load_disease_mapping(self):
        """Load disease mapping từ model_info hoặc tạo mới"""
        
        # Kiểm tra xem model_info có disease_mapping không
        if 'disease_mapping' in self.model_info:
            # Format mới: disease_mapping là dict {disease_id: disease_name}
            self.disease_mapping = self.model_info['disease_mapping']
            print(f"\n✅ Disease mapping loaded from model_info: {len(self.disease_mapping)} entries")
            
            # Tạo reverse mapping
            self.disease_name_to_id = {v: k for k, v in self.disease_mapping.items()}
            
        else:
            # Tạo mapping mặc định từ danh sách diseases
            print(f"\n⚠️  No disease_mapping in model_info. Creating default mapping...")
            diseases = self.model_info.get('diseases', [])
            
            # Tạo ID tự động: D001, D002, ...
            self.disease_mapping = {
                f"D{str(i+1).zfill(3)}": disease
                for i, disease in enumerate(sorted(diseases))
            }
            
            self.disease_name_to_id = {v: k for k, v in self.disease_mapping.items()}
            
            print(f"✅ Created default mapping: {len(self.disease_mapping)} diseases")
        
        # Debug: In ra 5 mapping đầu tiên
        print(f"\n📋 Sample disease mapping:")
        for i, (disease_id, disease_name) in enumerate(list(self.disease_mapping.items())[:5]):
            print(f"   {disease_id} -> {disease_name}")
    
    def extract_symptoms(self, content: str) -> List[str]:
        """
        Sử dụng PhoBERT NER để trích xuất triệu chứng từ text
        
        Args:
            content: Văn bản mô tả triệu chứng
            
        Returns:
            List các triệu chứng được nhận dạng
        """
        if self.phobert_model is None or self.phobert_tokenizer is None:
            raise ValueError("PhoBERT model chưa được load. Vui lòng gọi load_models() trước.")
        
        print(f"\n[NER] Input: {content}")
        
        # Tokenize
        inputs = self.phobert_tokenizer(
            content,
            return_tensors="pt",
            truncation=True,
            max_length=256,
            padding=False
        ).to(self.device)
        
        # Predict
        with torch.no_grad():
            outputs = self.phobert_model(**inputs)
            predictions = torch.argmax(outputs.logits, dim=-1)
        
        # Decode
        tokens = self.phobert_tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
        predicted_labels = [self.id2label[p.item()] for p in predictions[0]]
        
        # Extract symptoms (gộp B- và I- tags)
        symptoms = []
        current_symptom = []
        
        for token, label in zip(tokens, predicted_labels):
            # Skip special tokens
            if token in ['<s>', '</s>', '<pad>']:
                continue
            
            if label.startswith('B-SYMPTOM'):
                # Save previous symptom
                if current_symptom:
                    symptom_text = self._clean_tokens(current_symptom)
                    if symptom_text:
                        symptoms.append(symptom_text)
                
                # Start new symptom
                current_symptom = [token]
                
            elif label.startswith('I-SYMPTOM') and current_symptom:
                # Continue current symptom
                current_symptom.append(token)
                
            else:
                # End current symptom
                if current_symptom:
                    symptom_text = self._clean_tokens(current_symptom)
                    if symptom_text:
                        symptoms.append(symptom_text)
                    current_symptom = []
        
        # Save last symptom
        if current_symptom:
            symptom_text = self._clean_tokens(current_symptom)
            if symptom_text:
                symptoms.append(symptom_text)
        
        print(f"[NER] Extracted symptoms: {symptoms}")
        return symptoms
    
    def _clean_tokens(self, tokens: List[str]) -> str:
        """
        Làm sạch và ghép tokens thành text với format: thay khoảng trắng bằng dấu _
        
        PhoBERT tokenizer:
        - Token có suffix "@@" nghĩa là cần ghép liền với token tiếp theo
        - Ví dụ: ["số@@", "t@@"] -> "sốt" (ghép liền tất cả)
        - Ví dụ: ["đau", "đầu"] -> "đau_đầu" (hai từ riêng biệt)
        """
        if not tokens:
            return ""
        
        print(f"[DEBUG _clean_tokens] Input tokens: {tokens}")
        
        # Ghép các tokens, xử lý @@ suffix
        result = []
        buffer = ""  # Buffer để ghép các token liền nhau
        
        for i, token in enumerate(tokens):
            print(f"[DEBUG] Token {i}: '{token}'")
            
            if token.endswith("@@"):
                # Token này cần ghép liền với token tiếp theo
                clean_token = token[:-2]  # Remove @@ ở cuối
                # Remove leading underscore if exists
                if clean_token.startswith("_"):
                    clean_token = clean_token[1:]
                buffer += clean_token
                print(f"[DEBUG] Token ends with @@, adding to buffer: '{buffer}'")
            else:
                # Token bình thường hoặc token cuối cùng của chuỗi ghép
                clean_token = token
                # Remove leading underscore if exists
                if clean_token.startswith("_"):
                    clean_token = clean_token[1:]
                    
                buffer += clean_token
                print(f"[DEBUG] Completing word: '{buffer}'")
                
                if buffer:
                    result.append(buffer)
                    buffer = ""  # Reset buffer
        
        # Nếu còn buffer (trường hợp token cuối có @@)
        if buffer:
            result.append(buffer)
        
        print(f"[DEBUG _clean_tokens] After processing: {result}")
        
        # Join các tokens với dấu gạch dưới
        text = "_".join(result)
        
        print(f"[DEBUG _clean_tokens] Final result: '{text}'")
        return text.strip()
    
    def predict_disease(self, symptoms: List[str], top_k: int = 5) -> Dict[str, Any]:
        """
        Dự đoán bệnh từ danh sách triệu chứng
        
        Args:
            symptoms: List các triệu chứng
            top_k: Số lượng dự đoán hàng đầu trả về
            
        Returns:
            Dict chứa thông tin dự đoán
        """
        if self.bayes_model is None or self.model_info is None:
            raise ValueError("Bayes model chưa được load. Vui lòng gọi load_models() trước.")
        
        if not symptoms:
            raise ValueError("Danh sách triệu chứng không được rỗng")
        
        print(f"\n[BAYES] Input symptoms: {symptoms}")
        
        # Get all symptoms from model
        all_symptoms = self.model_info['symptoms']
        
        # Create feature vector
        feature_vector = np.zeros(len(all_symptoms))
        matched_symptoms = []
        unmatched_symptoms = []
        
        for symptom in symptoms:
            symptom_norm = symptom.lower().strip()
            matched = False
            
            # Tìm kiếm trong danh sách symptoms
            for idx, known_symptom in enumerate(all_symptoms):
                known_norm = known_symptom.lower().strip()
                
                # Exact match hoặc substring match
                if symptom_norm == known_norm or symptom_norm in known_norm or known_norm in symptom_norm:
                    feature_vector[idx] = 1
                    matched_symptoms.append(symptom)
                    matched = True
                    break
            
            if not matched:
                unmatched_symptoms.append(symptom)
        
        # Apply Laplace smoothing (như trong training)
        smoothing_value = 0.01
        feature_vector = feature_vector + smoothing_value
        feature_vector = feature_vector.reshape(1, -1)
        
        # Predict
        prediction = self.bayes_model.predict(feature_vector)[0]
        probabilities = self.bayes_model.predict_proba(feature_vector)[0]
        
        # Get top K predictions
        top_indices = np.argsort(probabilities)[::-1][:top_k]
        top_predictions = []
        
        for idx in top_indices:
            disease_name = self.bayes_model.classes_[idx]
            confidence = probabilities[idx]
            
            # Get disease_id
            disease_id = self.disease_name_to_id.get(disease_name, "UNKNOWN")
            
            top_predictions.append({
                "disease_id": disease_id,
                "disease_name": disease_name,
                "confidence": float(confidence)
            })
        
        # Main prediction
        main_prediction = top_predictions[0]
        
        print(f"[BAYES] Prediction: {main_prediction['disease_name']} ({main_prediction['disease_id']}) - {main_prediction['confidence']:.2%}")
        print(f"[BAYES] Matched: {len(matched_symptoms)}, Unmatched: {len(unmatched_symptoms)}")
        
        return {
            "disease_id": main_prediction["disease_id"],
            "disease_name": main_prediction["disease_name"],
            "confidence": main_prediction["confidence"],
            "matched_symptoms": matched_symptoms,
            "unmatched_symptoms": unmatched_symptoms,
            "top_predictions": top_predictions
        }