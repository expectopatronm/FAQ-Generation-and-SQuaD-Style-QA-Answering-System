import torch

from transformers import AutoTokenizer, AutoModelForSequenceClassification


class QuestionAnswerEvaluator:
    """Description."""

    def __init__(self):

        QAE_PRETRAINED = "iarfmoose/bert-base-cased-qa-evaluator"

        self.SEQ_LENGTH = 512
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.qae_tokenizer = AutoTokenizer.from_pretrained(QAE_PRETRAINED)
        self.qae_model = AutoModelForSequenceClassification.from_pretrained(
            QAE_PRETRAINED
        )
        self.qae_model.to(self.device)

    def encode_qa_pairs(self, questions, answers):
        encoded_pairs = []
        for i in range(len(questions)):
            encoded_qa = self.qae_tokenizer(
                text=questions[i],
                text_pair=answers[i],
                padding="max_length",
                max_length=self.SEQ_LENGTH,
                truncation=True,
                return_tensors="pt",
            )
            encoded_pairs.append(encoded_qa.to(self.device))
        return encoded_pairs

    def get_scores(self, encoded_qa_pairs):
        scores = {}
        self.qae_model.eval()
        with torch.no_grad():
            for i in range(len(encoded_qa_pairs)):
                scores[i] = self.qae_model(**encoded_qa_pairs[i])[0][0][1]
        return [
            k
            for k, v in sorted(
                scores.items(), key=lambda item: item[1], reverse=True
            )
        ]
