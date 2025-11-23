def merge_phobert_tokens(tokens):
    """
    Merge PhoBERT subword tokens correctly
    Example: ["số", "@@t"] -> "sốt"
    """
    if not tokens:
        return ""
    
    result = []
    current_word = ""
    
    for token in tokens:
        if token.startswith("@@"):
            # Subword token - merge với từ trước
            current_word += token.replace("@@", "")
        else:
            # Từ mới - lưu từ cũ và bắt đầu từ mới
            if current_word:
                result.append(current_word)
            current_word = token
    
    # Thêm từ cuối cùng
    if current_word:
        result.append(current_word)
    
    return " ".join(result)

def extract_entities_from_predictions(tokens, predictions, id2label):
    """
    Extract entities từ predictions của PhoBERT NER
    
    Returns:
        List[str]: Danh sách entities đã được merge đúng
    """
    entities = []
    current_entity = []
    current_entity_tokens = []
    
    for token, pred_id in zip(tokens, predictions):
        # Bỏ qua special tokens
        if token in ['<s>', '</s>', '<pad>']:
            continue
            
        label = id2label[pred_id]
        
        if label.startswith('B-'):
            # Bắt đầu entity mới - lưu entity cũ nếu có
            if current_entity_tokens:
                merged = merge_phobert_tokens(current_entity_tokens)
                entities.append(merged)
            
            current_entity_tokens = [token]
            
        elif label.startswith('I-') and current_entity_tokens:
            # Tiếp tục entity hiện tại
            current_entity_tokens.append(token)
            
        else:
            # Token 'O' - kết thúc entity
            if current_entity_tokens:
                merged = merge_phobert_tokens(current_entity_tokens)
                entities.append(merged)
                current_entity_tokens = []
    
    # Lưu entity cuối cùng
    if current_entity_tokens:
        merged = merge_phobert_tokens(current_entity_tokens)
        entities.append(merged)
    
    return entities