import re

class SymptomNormalizer:
    def __init__(self):
        # Mapping từ format merge → format gạch dưới
        self.symptom_map = {
            # Từ đơn
            "sốt": "sốt",
            "ho": "ho",
            
            # Cụm 2 từ
            "đau đầu": "đau_đầu",
            "ho khan": "ho_khan",
            "chảy nước mũi": "chảy_nước_mũi",
            "nghẹt mũi": "nghẹt_mũi",
            "đau họng": "đau_họng",
            "hắt hơi": "hắt_hơi",
            "mệt mỏi": "mệt_mỏi",
            
            # Cụm 3+ từ
            "đau đầu nhẹ": "đau_đầu_nhẹ",
            "đau cơ toàn thân": "đau_cơ_toàn_thân",
            "ho có đờm": "ho_có_đờm",
            "sốt cao": "sốt_cao",
            "buồn nôn": "buồn_nôn",
            
            # Thêm các mapping khác từ training data...
        }
        
        # Regex patterns cho matching linh hoạt
        self.patterns = self._build_patterns()
    
    def _build_patterns(self):
        """Tạo regex patterns cho fuzzy matching"""
        patterns = []
        for key in sorted(self.symptom_map.keys(), key=len, reverse=True):
            # Pattern cho phép khoảng trắng thừa
            pattern = re.escape(key).replace(r'\ ', r'\s+')
            patterns.append((re.compile(pattern, re.IGNORECASE), key))
        return patterns
    
    def normalize(self, symptom_text):
        """
        Chuẩn hóa triệu chứng về format gạch dưới
        
        Args:
            symptom_text (str): "đau đầu", "sốt cao", etc.
            
        Returns:
            str: "đau_đầu", "sốt_cao", etc.
        """
        # Chuẩn hóa whitespace
        text = re.sub(r'\s+', ' ', symptom_text.strip().lower())
        
        # Exact match
        if text in self.symptom_map:
            return self.symptom_map[text]
        
        # Fuzzy match
        for pattern, key in self.patterns:
            if pattern.search(text):
                return self.symptom_map[key]
        
        # Fallback: thay khoảng trắng bằng gạch dưới
        return text.replace(' ', '_')
    
    def normalize_list(self, symptoms):
        """Chuẩn hóa list triệu chứng"""
        return [self.normalize(s) for s in symptoms]
    
    @classmethod
    def from_training_data(cls, train_json_path):
        """
        Tự động build symptom_map từ training data
        """
        import json
        
        normalizer = cls()
        symptom_set = set()
        
        with open(train_json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        for item in data:
            tokens = item['tokens']
            labels = item['ner_tags']
            
            current_symptom = []
            for token, label in zip(tokens, labels):
                if label.startswith('B-SYMPTOM'):
                    if current_symptom:
                        symptom_set.add(' '.join(current_symptom))
                    current_symptom = [token]
                elif label.startswith('I-SYMPTOM'):
                    current_symptom.append(token)
                else:
                    if current_symptom:
                        symptom_set.add(' '.join(current_symptom))
                        current_symptom = []
            
            if current_symptom:
                symptom_set.add(' '.join(current_symptom))
        
        # Auto-generate mapping
        for symptom in symptom_set:
            normalized = symptom.replace(' ', '_')
            normalizer.symptom_map[symptom] = normalized
        
        return normalizer