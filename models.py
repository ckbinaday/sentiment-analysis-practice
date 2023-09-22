from transformers import AutoModelForSequenceClassification
from transformers import AutoTokenizer, AutoConfig
import numpy as np
from scipy.special import softmax


class TwitterRobertaBaseSentiment:
    """
    cardiffnlp/twitter-roberta-base-sentiment-latest - 2022
    https://huggingface.co/cardiffnlp/twitter-roberta-base-sentiment-latest
    """
    def __init__(self):
        self.model_name = f"cardiffnlp/twitter-roberta-base-sentiment-latest"
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.config = AutoConfig.from_pretrained(self.model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name)

    def saving_model_locally(self):
        config = AutoConfig.from_pretrained(self.model_name)
        self.tokenizer.save_pretrained(self.model_name)
        config.save_pretrained(self.model_name)
        self.model.save_pretrained(self.model_name)

    # Preprocess text (username and link placeholders)
    @staticmethod
    def preprocess(text):
        new_text = []
        for t in text.split(" "):
            t = '@user' if t.startswith('@') and len(t) > 1 else t
            t = 'http' if t.startswith('http') else t
            new_text.append(t)
        return " ".join(new_text)

    def predict(self, input_text: str):
        """
        :param input_text:
        :return:
        """
        text = self.preprocess(input_text)
        encoded_input = self.tokenizer(text, return_tensors='pt')
        output = self.model(**encoded_input)
        scores = output[0][0].detach().numpy()
        scores = softmax(scores)

        # Print labels and scores
        ranking = np.argsort(scores)
        ranking = ranking[::-1]
        # for i in range(scores.shape[0]):
        #     l = self.config.id2label[ranking[i]]
        #     s = scores[ranking[i]]
        #     print(f"{i + 1}) {l} {np.round(float(s), 4)}")

        return {"model_output": self.config.id2label[ranking[0]],
                "confidence_score": np.round(float(scores[ranking[0]]), 4) * 100}


if __name__ == '__main__':
    roberta = TwitterRobertaBaseSentiment()
    # roberta.saving_model_locally()
    print(roberta.predict('I hate war'))
