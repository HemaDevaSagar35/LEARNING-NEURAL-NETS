from datasets import load_dataset, laod_metric


def compute_metrics_multi(predictions, references):
    exact_matches = 0
    f1_scores = []
    
    for pred, refs in zip(predictions, references):
        # Check for an exact match with any reference
        exact_match = any(pred.strip().lower() == ref.strip().lower() for ref in refs)
        exact_matches += int(exact_match)
        
        # Compute F1 score for each reference and take the maximum
        f1_scores.append(max(compute_f1(pred, ref) for ref in refs))
    
    em_score = 100.0 * exact_matches / len(predictions)
    avg_f1_score = sum(f1_scores) / len(predictions)
    
    return {"exact_match": em_score, "f1": avg_f1_score}


def compute_f1(prediction, ground_truth):
    pred_tokens = prediction.strip().split()
    gt_tokens = ground_truth.strip().split()
    
    common_tokens = set(pred_tokens) & set(gt_tokens)
    if not common_tokens:
        return 0.0

    precision = len(common_tokens) / len(pred_tokens)
    recall = len(common_tokens) / len(gt_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1