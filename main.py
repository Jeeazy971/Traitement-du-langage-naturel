import pandas as pd
import re
import torch
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer
from transformers import Trainer, TrainingArguments
from transformers import BertForSequenceClassification

# Fonction pour nettoyer le texte


def clean_text(text):
    text = re.sub(r"http\S+|www\S+|https\S+", '', text,
                  flags=re.MULTILINE)  # supprimer les urls
    text = re.sub(r'\@\w+|\#', '', text)  # supprimer les mentions et hashtags
    return text


# Charger les données
train_data = pd.read_csv('train.txt', sep="\t",
                         header=None, names=["text", "emotion"])
val_data = pd.read_csv('val.txt', sep="\t", header=None,
                       names=["text", "emotion"])
test_data = pd.read_csv('test.txt', sep="\t",
                        header=None, names=["text", "emotion"])

# Fusionner les données d'entraînement et de validation
data = pd.concat([train_data, val_data])

# Transformer les émotions en entiers
emotion_dict = {emotion: i for i, emotion in enumerate(data.emotion.unique())}
data['emotion'] = data['emotion'].map(emotion_dict)
# Assurez-vous que les mêmes émotions sont présentes dans l'ensemble de test
test_data['emotion'] = test_data['emotion'].map(emotion_dict)

# Nettoyer le texte
data['text'] = data['text'].apply(clean_text)
# Appliquer le nettoyage aussi à l'ensemble de test
test_data['text'] = test_data['text'].apply(clean_text)

# Diviser les données en entraînement et test
train_texts, test_texts, train_labels, test_labels = train_test_split(
    data.text, data.emotion, test_size=0.2)

# Initialiser le tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Tokeniser les données
train_encodings = tokenizer(train_texts.to_list(),
                            truncation=True, padding=True)
test_encodings = tokenizer(test_texts.to_list(), truncation=True, padding=True)


class EmotionDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx])
                for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(
            self.labels[idx]).float()  # Convert labels to float
        return item

    def __len__(self):
        return len(self.labels)


# Créer le jeu de données
train_dataset = EmotionDataset(train_encodings, train_labels.tolist())
test_dataset = EmotionDataset(test_encodings, test_labels.tolist())

# Initialiser le modèle
model = BertForSequenceClassification.from_pretrained(
    'bert-base-uncased', num_labels=len(emotion_dict))

# Entraîner le modèle
# Définir les arguments d'entraînement
training_args = TrainingArguments(
    output_dir='./results',          # les résultats seront stockés ici
    num_train_epochs=3,              # nombre total d'époques d'entraînement
    per_device_train_batch_size=16,  # taille du batch pour l'entraînement
    per_device_eval_batch_size=64,   # taille du batch pour l'évaluation
    warmup_steps=500,                # nombre de steps pour le warmup
    weight_decay=0.01,               # decay du poids
    logging_dir='./logs',            # directory pour les logs
)

# Initialiser le Trainer
trainer = Trainer(
    model=model,                         # le modèle à entraîner
    args=training_args,                  # les arguments d'entraînement
    train_dataset=train_dataset,         # jeu de données d'entraînement
    eval_dataset=test_dataset            # jeu de données d'évaluation
)

# Entraîner le modèle
trainer.train()

# Évaluer le modèle
eval_results = trainer.evaluate()

print(eval_results)

# Prédire l'émotion d'un nouveau texte
text = "Je suis très heureux aujourd'hui !"
encoding = tokenizer(text, truncation=True, padding=True, return_tensors='pt')
logits = model(**encoding)[0]
predicted_emotion = torch.argmax(logits).item()

# Transformer l'émotion prédite en texte
inverse_emotion_dict = {i: emotion for emotion, i in emotion_dict.items()}
predicted_emotion_text = inverse_emotion_dict[predicted_emotion]
print(f"L'émotion prédite est : {predicted_emotion_text}")
