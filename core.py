from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification
)

import torch

class PsychosisModel:
    def __init__(self, pretrained_model):
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model, max_length=512)
        self.model = AutoModelForSequenceClassification.from_pretrained(pretrained_model)

    def predict(self, data):
        inputs = self.tokenizer(data.text, return_tensors="pt")
        with torch.no_grad():
            logits = self.model(**inputs).logits
            predicted_class_id = logits.argmax().item()

        return {
            'topic': data.topic,
            'predicted_class': predicted_class_id,
            'predicted_label': self.model.config.id2label[predicted_class_id],
            'probabilities': self.get_probabilites(logits),
        }
    
    def get_probabilites(self, logits):
        labels = self.model.config.id2label
        probabilities = torch.nn.functional.softmax(logits, dim=1)
        probabilities = probabilities.cpu().detach().numpy()[0]
        label_probabilities = [{'label': labels[i], 'score': probabilities[i].item()} for i in range(len(probabilities))]

        return sorted(label_probabilities, key=lambda x: x['score'], reverse=True)
