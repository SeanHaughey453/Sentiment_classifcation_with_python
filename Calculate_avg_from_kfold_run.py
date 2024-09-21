
reports = [
    {'negative': {'precision': 0.6946107784431138, 'recall': 0.6170212765957447, 'f1-score': 0.6535211267605634}, 'neutral': {'precision': 0.25196850393700787, 'recall': 0.3047619047619048, 'f1-score': 0.27586206896551724}, 'positive': {'precision': 0.8668555240793201, 'recall': 0.8656294200848657, 'f1-score': 0.8662420382165605}, 'accuracy': 0.76, 'macro avg': {'precision': 0.6044782688198139, 'recall': 0.5958042004808384, 'f1-score': 0.5985417446475471}, 'weighted avg': {'precision': 0.7699103747847705, 'recall': 0.76, 'f1-score': 0.7642606100914735}},
    {'negative': {'precision': 0.6904761904761905, 'recall': 0.5888324873096447, 'f1-score': 0.6356164383561644}, 'neutral': {'precision': 0.25892857142857145, 'recall': 0.22137404580152673, 'f1-score': 0.23868312757201646}, 'positive': {'precision': 0.8375, 'recall': 0.8973214285714286, 'f1-score': 0.8663793103448276}, 'accuracy': 0.748, 'macro avg': {'precision': 0.5956349206349206, 'recall': 0.5691759872275334, 'f1-score': 0.5802262920910028}, 'weighted avg': {'precision': 0.7327434523809525, 'recall': 0.748, 'f1-score': 0.7386908246198227}},
    {'negative': {'precision': 0.672514619883041, 'recall': 0.6149732620320856, 'f1-score': 0.6424581005586593}, 'neutral': {'precision': 0.34782608695652173, 'recall': 0.39344262295081966, 'f1-score': 0.36923076923076925}, 'positive': {'precision': 0.8813314037626628, 'recall': 0.8813314037626628, 'f1-score': 0.8813314037626628}, 'accuracy': 0.772, 'macro avg': {'precision': 0.6338907035340752, 'recall': 0.6299157629151894, 'f1-score': 0.631006757850697}, 'weighted avg': {'precision': 0.7771950165268242, 'recall': 0.772, 'f1-score': 0.7741858186506231}},
    {'negative': {'precision': 0.7241379310344828, 'recall': 0.5585106382978723, 'f1-score': 0.6306306306306306}, 'neutral': {'precision': 0.2366412213740458, 'recall': 0.28440366972477066, 'f1-score': 0.25833333333333336}, 'positive': {'precision': 0.8701657458563536, 'recall': 0.8961593172119487, 'f1-score': 0.8829712683952348}, 'accuracy': 0.766, 'macro avg': {'precision': 0.610314966088294, 'recall': 0.5796912084115305, 'f1-score': 0.5906450774530662}, 'weighted avg': {'precision': 0.7736583435012704, 'recall': 0.766, 'f1-score': 0.767445693573742}},
    {'negative': {'precision': 0.654639175257732, 'recall': 0.7055555555555556, 'f1-score': 0.679144385026738}, 'neutral': {'precision': 0.2112676056338028, 'recall': 0.23809523809523808, 'f1-score': 0.22388059701492538}, 'positive': {'precision': 0.8674698795180723, 'recall': 0.829971181556196, 'f1-score': 0.8483063328424153}, 'accuracy': 0.733, 'macro avg': {'precision': 0.5777922201365358, 'recall': 0.5912073250689965, 'f1-score': 0.5837771049613596}, 'weighted avg': {'precision': 0.746478866241793, 'recall': 0.733, 'f1-score': 0.7391795395213296}}
]

metrics = {
    'negative': {'precision': 0, 'recall': 0, 'f1-score': 0},
    'neutral': {'precision': 0, 'recall': 0, 'f1-score': 0},
    'positive': {'precision': 0, 'recall': 0, 'f1-score': 0},
    'accuracy': 0,
    'macro avg': {'precision': 0, 'recall': 0, 'f1-score': 0},
    'weighted avg': {'precision': 0, 'recall': 0, 'f1-score': 0}
}

for report in reports:
    for key in metrics.keys():
        if key == 'accuracy':
            metrics[key] += report[key]
        else:
            metrics[key]['precision'] += report[key]['precision']
            metrics[key]['recall'] += report[key]['recall']
            metrics[key]['f1-score'] += report[key]['f1-score']

num_folds = len(reports)
avg_metrics = {k: {} for k in metrics}
for key, metrics in metrics.items():
    if key == 'accuracy':
        avg_metrics[key] = metrics / num_folds
    else:
        avg_metrics[key]['precision'] = metrics['precision'] / num_folds
        avg_metrics[key]['recall'] = metrics['recall'] / num_folds
        avg_metrics[key]['f1-score'] = metrics['f1-score'] / num_folds

for key, metrics in avg_metrics.items():
    if key == 'accuracy':
        print(f"Average {key}: {metrics:.3f}")
    else:
        print(f"Average {key} - Precision: {metrics['precision']:.3f}, Recall: {metrics['recall']:.3f}, F1-score: {metrics['f1-score']:.3f}")
