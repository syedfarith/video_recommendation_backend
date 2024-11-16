def evaluate_recommendations(recommended, actual):
    hits = len(set(recommended) & set(actual))
    precision = hits / len(recommended)
    recall = hits / len(actual)
    return {"precision": precision, "recall": recall}

def calculate_ctr(clicks, views):
    return clicks / views if views > 0 else 0
