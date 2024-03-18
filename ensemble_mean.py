import copy

def media_ponderada(valor1, peso1, valor2, peso2):
    total_pesos = peso1 + peso2
    media_ponderada = (peso1 * valor1 + peso2 * valor2) / total_pesos
    return media_ponderada

def align(prediction1, prediction2, weights, labels):
    labels_p1 = labels[0]
    labels_p2 = labels[1]
    ensemble_pred = copy.deepcopy(prediction1)
    for label in labels_p2:
        idx = labels_p1.index(label)
        pred1 = prediction1[idx]
        pred2 = prediction2[labels_p2.index(label)]
        avg = media_ponderada(pred1, weights[0], pred2, weights[1])
        ensemble_pred[idx] = avg
    return ensemble_pred

def ensemble_by_weighted_mean(models, weights, labels):
    model_1 = models[0]
    model_2 = models[1]
    return align(model_1, model_2, weights, labels)

