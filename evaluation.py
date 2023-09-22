import json
import pandas as pd
from models import TwitterRobertaBaseSentiment
from sklearn.metrics import precision_recall_fscore_support, accuracy_score


def split_json_model_output(model_output):
    return model_output['model_output'], model_output['confidence_score']


if __name__ == '__main__':
    df_evaluation = pd.read_csv('sentiment_test_cases.csv')

    # Load Model
    twitter_roberta = TwitterRobertaBaseSentiment()

    # Prediction
    df_evaluation['model_output'], df_evaluation['confidence_score'] = zip(
        *df_evaluation['text'].apply(lambda x: split_json_model_output(twitter_roberta.predict(x))))

    # Overall Performance
    y_true = df_evaluation['expected_sentiment'].tolist()
    y_pred = df_evaluation['model_output'].tolist()
    accuracy = accuracy_score(y_true, y_pred)
    macro_pr, macro_re, macro_f1, _ = precision_recall_fscore_support(y_true, y_pred, average='macro')
    micro_pr, micro_re, micro_f1, _ = precision_recall_fscore_support(y_true, y_pred, average='micro')
    weighted_pr, weighted_re, weighted_f1, _ = precision_recall_fscore_support(y_true, y_pred, average='weighted')

    performance = {
        'Total Data Points': len(df_evaluation),
        'Matched': int(df_evaluation.apply(
            lambda x: 1 if x['expected_sentiment'] == x['model_output'] else 0, axis=1).sum()),
        'Accuracy': accuracy,
        'Macro Precision': macro_pr,
        'Macro Recall': macro_re,
        'Macro F1-score': macro_f1,
        'Micro Precision': micro_pr,
        'Micro Recall': micro_re,
        'Micro F1-score': micro_f1,
        'Weighted Precision': weighted_pr,
        'Weighted Recall': weighted_re,
        'Weighted F1-score': weighted_f1
    }
    print(performance)

    # Save
    df_evaluation.to_csv('output_sentiment_test.csv')
    with open('TwitterRobertaBaseSentiment_performance.json', 'w') as f:
        json.dump(performance, f)
