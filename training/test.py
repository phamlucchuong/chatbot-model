"""
Script test 2 models: NER và Naive Bayes
- NER model: Trích xuất triệu chứng từ câu
- Naive Bayes model: Dự đoán bệnh dựa trên triệu chứng
"""

import spacy
import pickle
import json
import numpy as np
from pathlib import Path

# Đường dẫn đến models
NER_MODEL_PATH = "models/spacy_ner_model"
BAYES_MODEL_PATH = "models/naive_bayes_model.pkl"
MODEL_INFO_PATH = "models/model_info.json"

def load_models():
    """Load cả 2 models và thông tin cần thiết"""
    print("📂 Đang load models...")
    
    # Load NER model (spaCy)
    try:
        nlp = spacy.load(NER_MODEL_PATH)
        print(f"✓ Đã load NER model từ {NER_MODEL_PATH}")
    except Exception as e:
        print(f"❌ Lỗi khi load NER model: {e}")
        return None, None, None
    
    # Load Naive Bayes model
    try:
        with open(BAYES_MODEL_PATH, "rb") as f:
            bayes_model = pickle.load(f)
        print(f"✓ Đã load Naive Bayes model từ {BAYES_MODEL_PATH}")
    except Exception as e:
        print(f"❌ Lỗi khi load Naive Bayes model: {e}")
        return None, None, None
    
    # Load model info
    try:
        with open(MODEL_INFO_PATH, "r", encoding="utf-8") as f:
            model_info = json.load(f)
        print(f"✓ Đã load model info từ {MODEL_INFO_PATH}")
    except Exception as e:
        print(f"❌ Lỗi khi load model info: {e}")
        return None, None, None
    
    print()
    return nlp, bayes_model, model_info


def extract_symptoms_ner(text, nlp_model):
    """
    Trích xuất triệu chứng từ câu sử dụng NER model
    Trả về danh sách các triệu chứng đã được normalize
    """
    doc = nlp_model(text)
    symptoms = []
    
    # Lấy các entity có label là SYMPTOM
    for ent in doc.ents:
        if ent.label_ == "SYMPTOM":
            symptom_text = ent.text.lower().strip()
            # Normalize: thay khoảng trắng bằng dấu gạch dưới
            symptom_normalized = symptom_text.replace(" ", "_")
            symptoms.append(symptom_normalized)
    
    return symptoms


def normalize_symptom_name(symptom, all_symptoms):
    """
    Normalize tên triệu chứng để khớp với format trong model
    Có thể có nhiều biến thể của cùng một triệu chứng
    """
    symptom_lower = symptom.lower().strip().replace(" ", "_")
    
    # Kiểm tra exact match trước
    if symptom_lower in all_symptoms:
        return symptom_lower
    
    # Nếu không tìm thấy, tìm partial match
    for s in all_symptoms:
        if symptom_lower in s or s in symptom_lower:
            return s
    
    # Nếu vẫn không tìm thấy, trả về triệu chứng gốc (đã normalize)
    return symptom_lower


def predict_disease_bayes(symptoms_list, bayes_model, model_info):
    """
    Dự đoán bệnh sử dụng Naive Bayes model
    """
    all_symptoms = model_info["symptoms"]
    
    # Normalize các triệu chứng
    normalized_symptoms = [
        normalize_symptom_name(s, all_symptoms) 
        for s in symptoms_list
    ]
    
    # Tạo binary vector
    vector = [1 if symptom in normalized_symptoms else 0 for symptom in all_symptoms]
    vector = np.array(vector).reshape(1, -1)
    
    # Dự đoán
    prediction = bayes_model.predict(vector)[0]
    probabilities = bayes_model.predict_proba(vector)[0]
    
    # Lấy top 3 bệnh có xác suất cao nhất
    top_indices = np.argsort(probabilities)[::-1][:3]
    top_diseases = [
        (bayes_model.classes_[i], probabilities[i]) 
        for i in top_indices
    ]
    
    return prediction, top_diseases, normalized_symptoms


def test_combined_models(text, nlp_model, bayes_model, model_info):
    """
    Test kết hợp cả 2 models:
    1. NER model trích xuất triệu chứng
    2. Naive Bayes dự đoán bệnh dựa trên triệu chứng
    """
    print(f"\n{'='*70}")
    print(f"📝 Input: {text}")
    print(f"{'='*70}")
    
    # Bước 1: Trích xuất triệu chứng bằng NER
    symptoms = extract_symptoms_ner(text, nlp_model)
    
    if not symptoms:
        print("⚠️  NER model không tìm thấy triệu chứng nào")
        print("   → Không thể dự đoán bệnh")
        return None, None, []
    
    print(f"\n🔍 NER Model - Triệu chứng tìm được ({len(symptoms)}):")
    for i, symptom in enumerate(symptoms, 1):
        print(f"   {i}. {symptom}")
    
    # Bước 2: Dự đoán bệnh bằng Naive Bayes
    try:
        prediction, top_diseases, normalized_symptoms = predict_disease_bayes(
            symptoms, bayes_model, model_info
        )
        
        print(f"\n🏥 Naive Bayes Model - Chẩn đoán:")
        print(f"   ➜ Bệnh dự đoán: {prediction}")
        
        print(f"\n📊 Top 3 khả năng:")
        for i, (disease, prob) in enumerate(top_diseases, 1):
            print(f"   {i}. {disease}: {prob:.2%}")
        
        print(f"\n💡 Triệu chứng đã normalize:")
        for i, symptom in enumerate(normalized_symptoms, 1):
            is_found = symptom in model_info["symptoms"]
            status = "✓" if is_found else "⚠️ (không có trong model)"
            print(f"   {i}. {symptom} {status}")
        
        return prediction, top_diseases, symptoms
        
    except Exception as e:
        print(f"\n❌ Lỗi khi dự đoán bệnh: {e}")
        return None, None, symptoms


def main():
    """Hàm main để test các model"""
    print("="*70)
    print("🧪 TEST COMBINED MODELS: NER + NAIVE BAYES")
    print("="*70)
    print()
    
    # Load models
    nlp_model, bayes_model, model_info = load_models()
    
    if nlp_model is None or bayes_model is None or model_info is None:
        print("\n❌ Không thể load models. Vui lòng kiểm tra lại đường dẫn.")
        return
    
    print(f"📊 Thông tin models:")
    print(f"   - NER labels: {[label for label in nlp_model.get_pipe('ner').labels]}")
    print(f"   - Naive Bayes classes: {len(bayes_model.classes_)} bệnh")
    print(f"   - Symptoms trong model: {len(model_info['symptoms'])} triệu chứng")
    print()
    
    # Test cases
    test_cases = [
        "Tôi bị sốt cao và đau đầu dữ dội",
        "Mấy ngày nay tôi ho khan và đau họng",
        "Tôi bị sổ mũi và nghẹt mũi",
        "Tôi có triệu chứng tiêu chảy và đau bụng quặn",
        "Tôi bị đau nhức toàn thân, sốt và ho",
        "Tôi bị ngứa da, nổi mẩn đỏ và bong vảy",
        "Tôi bị đau họng, khó nuốt và sốt nhẹ",
        "Tôi bị đau bụng dưới và tiểu buốt",
        "Tôi bị đau đầu một bên và nhạy cảm ánh sáng",
        "Tôi bị sốt, ho có đờm, đau bụng dữ dội, chóng mặt và nôn mửa",
    ]
    
    results = []
    
    # Test từng câu
    for i, text in enumerate(test_cases, 1):
        print(f"\n{'#'*70}")
        print(f"# TEST CASE {i}/{len(test_cases)}")
        print(f"{'#'*70}")
        
        prediction, top_diseases, symptoms = test_combined_models(
            text, nlp_model, bayes_model, model_info
        )
        
        results.append({
            "text": text,
            "symptoms": symptoms,
            "prediction": prediction,
            "top_diseases": top_diseases
        })

    print("Biện pháp sơ cứu: sục 10 lần 1 ngày, kiên trì trong 4 tuần để bệnh tình thuyên giảm")
    
    # Tổng kết
    print(f"\n\n{'='*70}")
    print("📊 TỔNG KẾT")
    print(f"{'='*70}")
    print(f"✓ Đã test {len(test_cases)} câu")
    print(f"✓ Số câu tìm thấy triệu chứng: {sum(1 for r in results if r['symptoms'])}")
    print(f"✓ Số câu dự đoán được bệnh: {sum(1 for r in results if r['prediction'])}")
    
    print(f"\n{'='*70}")
    print("✅ Hoàn tất test!")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()

