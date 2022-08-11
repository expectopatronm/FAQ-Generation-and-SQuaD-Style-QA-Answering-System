import torch

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


class QuestionGenerator:
    """Description."""

    def __init__(self):

        QG_PRETRAINED = "iarfmoose/t5-base-question-generator"

        self.ANSWER_TOKEN = "<answer>"
        self.CONTEXT_TOKEN = "<context>"
        self.SEQ_LENGTH = 512
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        self.qg_tokenizer = AutoTokenizer.from_pretrained(
            QG_PRETRAINED, use_fast=False
        )
        self.qg_model = AutoModelForSeq2SeqLM.from_pretrained(QG_PRETRAINED)
        self.qg_model.to(self.device)
        self.qg_model.eval()

    def generate_questions(self, answers, passage):
        questions = []

        for ans in answers:
            qg_input = "{} {} {} {}".format(
                self.ANSWER_TOKEN, ans, self.CONTEXT_TOKEN, passage
            )

            encoded_input = self.qg_tokenizer(
                qg_input,
                padding="max_length",
                max_length=self.SEQ_LENGTH,
                truncation=True,
                return_tensors="pt",
            ).to(self.device)
            with torch.no_grad():
                output = self.qg_model.generate(
                    input_ids=encoded_input["input_ids"]
                )
            question = self.qg_tokenizer.decode(
                output[0], skip_special_tokens=True
            )
            questions.append(question)
        return list(questions)
