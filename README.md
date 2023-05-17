# Traitement du Langage Naturel : Étiquetage Automatique des Émotions d'un Texte

Ce projet utilise le modèle BertForSequenceClassification du package `transformers` pour prédire l'émotion contenue dans un texte. Les données sont chargées à partir de fichiers textes, nettoyées et transformées en encodages que le modèle Bert peut comprendre. Le modèle est ensuite entraîné et évalué sur ces données.

## Comment utiliser ce projet

### 1. Installer les dépendances
Avant de pouvoir utiliser ce projet, vous devez installer certaines dépendances. Vous pouvez le faire en utilisant pip:

```bash
pip install pandas
pip install regex
pip install torch
pip install sklearn
pip install transformers
```

### 2. Importer les bibliothèques nécessaires

```python
import pandas as pd
import re
import torch
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer
from transformers import Trainer, TrainingArguments
from transformers import BertForSequenceClassification
```

### 3. Nettoyage du texte
Définissez une fonction pour nettoyer le texte. Cette fonction supprime les URL, les mentions et les hashtags.

```python
def clean_text(text):
    text = re.sub(r"http\S+|www\S+|https\S+", '', text,
                  flags=re.MULTILINE)  # supprimer les urls
    text = re.sub(r'\@\w+|\#', '', text)  # supprimer les mentions et hashtags
    return text
```

### 4. Chargement des données
Chargez les données à partir de fichiers textes. Les données sont séparées en trois ensembles : entraînement, validation et test. Fusionnez les ensembles d'entraînement et de validation pour ce projet.

```python
train_data = pd.read_csv('train.txt', sep="\t",
                         header=None, names=["text", "emotion"])
val_data = pd.read_csv('val.txt', sep="\t", header=None,
                       names=["text", "emotion"])
test_data = pd.read_csv('test.txt', sep="\t",
                        header=None, names=["text", "emotion"])

# Fusionner les données d'entraînement et de validation
data = pd.concat([train_data, val_data])
```

### 5. Préparation des données
Transformez les émotions en entiers et nettoyez le texte. Divisez ensuite les données en ensembles d'entraînement et de test.

```python
# Transformer les émotions en entiers
emotion_dict = {emotion: i for i, emotion in enumerate(data.emotion.unique())}
data['emotion'] = data['emotion'].map(emotion_dict)

# Nettoyer le texte
data['text'] = data['text'].apply(clean_text)

# Diviser les données en entraînement et test
train_texts, test_texts, train_labels, test_labels = train_test_split(
    data.text, data.emotion, test_size=0.2)
```

### 6. Initialisation du tokenizer
Utilisez le tokenizer `BertTokenizer` pour transformer vos textes en encodages que le modèle Bert peut comprendre.

```python
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
```

### 7. Tokenisation des données
Utilisez le tokenizer pour transformer vos textes en encodages.

```python
train_encodings = tokenizer(train_texts.to_list(),
                            truncation=True, padding=True)
test_encodings = tokenizer(test_texts.to_list

(), truncation=True, padding=True)
```

### 8. Création du jeu de données
Créez un jeu de données à partir de vos encodages.

```python
class EmotionDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx])
                for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(
            self.labels[idx]).float()
        return item

    def __len__(self):
        return len(self.labels)

train_dataset = EmotionDataset(train_encodings, train_labels.tolist())
test_dataset = EmotionDataset(test_encodings, test_labels.tolist())
```

### 9. Initialisation du modèle
Initialisez le modèle `BertForSequenceClassification`.

```python
model = BertForSequenceClassification.from_pretrained(
    'bert-base-uncased', num_labels=len(emotion_dict))
```

### 10. Entraînement du modèle
Utilisez la classe `Trainer` pour entraîner votre modèle.

```python
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=3,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=64,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir='./logs',
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset
)

trainer.train()
```

### 11. Évaluation du modèle
Évaluez votre modèle sur l'ensemble de test.

```python
eval_results = trainer.evaluate()
```

### 12. Prédiction de l'émotion
Prévoyez l'émotion d'un nouveau texte.

```python
text = "Je suis très heureux aujourd'hui !"
encoding = tokenizer(text, truncation=True, padding=True, return_tensors='pt')
logits = model(**encoding)[0]
predicted_emotion = torch.argmax(logits).item()

inverse_emotion_dict = {i: emotion for emotion, i in emotion_dict.items()}
predicted_emotion_text = inverse_emotion_dict[predicted_emotion]
print(f"L'émotion prédite est : {predicted_emotion_text}")
```
 Bonne programmation !
