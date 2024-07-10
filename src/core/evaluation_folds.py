import statistics

def compute_folds_metrics(fold_scores: dict) -> dict:
    # Calculate mean metrics across folds
    mean_metrics = {}
    std_metrics = {}

    for metric in fold_scores['fold_0'].keys():
        metric_values = [d[metric] for d in fold_scores.values()]
        mean_metrics[metric] = sum(metric_values) / len(metric_values)
        std_metrics[f'{metric}_std'] = statistics.stdev(metric_values)

    combined_dict = {**fold_scores, **mean_metrics, **std_metrics}

    return combined_dict
