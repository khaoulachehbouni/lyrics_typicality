import numpy as np
import pandas as pd
from fast_ml.model_development import train_valid_test_split
from transformers import Trainer, TrainingArguments, AutoConfig, AutoTokenizer, AutoModelForSequenceClassification
import torch
from torch import nn
from torch.nn.functional import softmax
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder
import datasets


#MoodyLyrics: A Sentiment Annotated Lyrics Dataset 
sent_w_lyrics = pd.read_csv("/content/sent_w_lyrics.csv")


#Create 2 binary variables valence and arousal
sent_w_lyrics['valence'] = np.where((sent_w_lyrics['Mood']=='relaxed') | (sent_w_lyrics['Mood']=='happy'),1,0)
sent_w_lyrics['arousal'] = np.where((sent_w_lyrics['Mood']=='angry') | (sent_w_lyrics['Mood']=='happy'),1,0)



#The code used for sentiment classification comes from: https://towardsdatascience.com/how-to-train-a-deep-learning-sentiment-analysis-model-4716c946c2ea
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print (f'Device Availble: {DEVICE}')

df_sent_valence = sent_w_lyrics[['lyrics_w_puncta', 'valence']]
df_sent_arousal = sent_w_lyrics[['lyrics_w_puncta', 'arousal']]



#Transform labels 
le_val = LabelEncoder()
df_sent_valence['valence'] = le_val.fit_transform(df_sent_valence['valence'])
df_sent_valence.head()

#Transform labels to integers
le_aro = LabelEncoder()
df_sent_arousal['arousal'] = le_aro.fit_transform(df_sent_arousal['arousal'])
df_sent_arousal.head()



#Split data into train, validation and test set : Valence
(train_texts_valence, train_labels_valence,
 val_texts_valence, val_labels_valence,
 test_texts_valence, test_labels_valence) = train_valid_test_split(df_sent_valence, target = 'valence', method='sorted', sort_by_col='lyrics_w_puncta', train_size=0.8, valid_size=0.1, test_size=0.1)


#Split data into train, validation and test set : Arousal
(train_texts_arousal, train_labels_arousal,
 val_texts_arousal, val_labels_arousal,
 test_texts_arousal, test_labels_arousal) = train_valid_test_split(df_sent_arousal, target = 'arousal', method='sorted', sort_by_col='lyrics_w_puncta', train_size=0.8, valid_size=0.1, test_size=0.1)


#Convert the review text from pandas series to list of sentences : Valence
train_texts_valence = train_texts_valence['lyrics_w_puncta'].to_list()
train_labels_valence = train_labels_valence.to_list()
val_texts_valence = val_texts_valence['lyrics_w_puncta'].to_list()
val_labels_valence = val_labels_valence.to_list()
test_texts_valence = test_texts_valence['lyrics_w_puncta'].to_list()
test_labels_valence = test_labels_valence.to_list()

#Convert the review text from pandas series to list of sentences : Arousal
train_texts_arousal = train_texts_arousal['lyrics_w_puncta'].to_list()
train_labels_arousal = train_labels_arousal.to_list()
val_texts_arousal = val_texts_arousal['lyrics_w_puncta'].to_list()
val_labels_arousal = val_labels_arousal.to_list()
test_texts_arousal = test_texts_arousal['lyrics_w_puncta'].to_list()
test_labels_arousal = test_labels_arousal.to_list()


#Create a DataLoader class for processing and loading of the data during training and inference phase
class DataLoader(torch.utils.data.Dataset):
    def __init__(self, sentences=None, labels=None):
        self.sentences = sentences
        self.labels = labels
        self.tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')
        
        if bool(sentences):
            self.encodings = self.tokenizer(self.sentences,
                                            truncation = True,
                                            padding = True)
        
    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        
        if self.labels == None:
            item['labels'] = None
        else:
            item['labels'] = torch.tensor(self.labels[idx])
        return item
    def __len__(self):
        return len(self.sentences)
    
    
    def encode(self, x):
        return self.tokenizer(x, return_tensors = 'pt').to(DEVICE)
    

#VALENCE
train_dataset_valence = DataLoader(train_texts_valence, train_labels_valence)
val_dataset_valence = DataLoader(val_texts_valence, val_labels_valence)
test_dataset_valence = DataLoader(test_texts_valence, test_labels_valence)

#AROUSAL
train_dataset_arousal = DataLoader(train_texts_arousal, train_labels_arousal)
val_dataset_arousal = DataLoader(val_texts_arousal, val_labels_arousal)
test_dataset_arousal = DataLoader(test_texts_arousal, test_labels_arousal)



#Define evaluation metrics
f1 = datasets.load_metric('f1')
accuracy = datasets.load_metric('accuracy')
precision = datasets.load_metric('precision')
recall = datasets.load_metric('recall')
def compute_metrics(eval_pred):
    metrics_dict = {}
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    
    metrics_dict.update(f1.compute(predictions = predictions, references = labels, average = 'macro'))
    metrics_dict.update(accuracy.compute(predictions = predictions, references = labels))
    metrics_dict.update(precision.compute(predictions = predictions, references = labels, average = 'macro'))
    metrics_dict.update(recall.compute(predictions = predictions, references = labels, average = 'macro'))
    return metrics_dict



#We configure instantiate a distilbert-base-uncased model from pretrained checkpoint.
config = AutoConfig.from_pretrained('distilbert-base-uncased',
                                    num_labels = 2)
model = AutoModelForSequenceClassification.from_config(config)



#VALENCE
training_args_val = TrainingArguments(
    num_train_epochs=10,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    warmup_steps=500,
    weight_decay=0.05,
    report_to='none',
    evaluation_strategy='steps',
    logging_steps=50)

trainer_val = Trainer(
    model=model,
    args=training_args_val,
    train_dataset=train_dataset_valence,
    eval_dataset=val_dataset_valence,
    compute_metrics=compute_metrics)

trainer_val.train()
eval_results_val = trainer_val.predict(test_dataset_valence)

id2label = {idx:label for idx, label in enumerate(le_val.classes_)}
label2id_mapper = id2label
proba = softmax(torch.from_numpy(eval_results_val.predictions))
pred = [label2id_mapper[i] for i in torch.argmax(proba, dim = -1).numpy()]
actual = [label2id_mapper[i] for i in eval_results_val.label_ids]

valence_df = pd.DataFrame({"lyrics":test_texts_valence, "actual":actual,"predicted":pred})
valence_df.head()

trainer_val.save_model('/content/sentiment_model_valence.pt')


#AROUSAL
training_args_aro = TrainingArguments(
    num_train_epochs=10,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    warmup_steps=500,
    weight_decay=0.05,
    report_to='none',
    evaluation_strategy='steps',
    logging_steps=50)

trainer_aro = Trainer(
    model=model,
    args=training_args_aro,
    train_dataset=train_dataset_arousal,
    eval_dataset=val_dataset_arousal,
    compute_metrics=compute_metrics)

trainer_aro.train()
eval_results_aro = trainer_aro.predict(test_dataset_arousal)

label2id_mapper = id2label
proba = softmax(torch.from_numpy(eval_results_aro.predictions))
pred = [label2id_mapper[i] for i in torch.argmax(proba, dim = -1).numpy()]
actual = [label2id_mapper[i] for i in eval_results_aro.label_ids]

arousal_df = pd.DataFrame({"lyrics":test_texts_arousal, "actual_arousal":actual,"predicted_arousal":pred})
arousal_df.head()

trainer_val.save_model('/content/sentiment_model_arousal.pt')


################################################INFERENCE ON OUR DATA
class SentimentModel():
    
    def __init__(self, model_path):
        
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path).to(DEVICE)
        args =  TrainingArguments(output_dir='/kaggle/working/results', per_device_eval_batch_size=64)
        self.batch_model = Trainer(model = self.model, args= args)
        self.single_dataloader = DataLoader()
        
    def batch_predict_proba(self, x):
        
        predictions = self.batch_model.predict(DataLoader(x))
        logits = torch.from_numpy(predictions.predictions)
        
        if DEVICE == 'cpu':
            proba = torch.nn.functional.softmax(logits, dim = 1).detach().numpy()
        else:
            proba = torch.nn.functional.softmax(logits, dim = 1).to('cpu').detach().numpy()
        return proba
        
        
    def predict_proba(self, x):
        
        x = self.single_dataloader.encode(x).to(DEVICE)
        predictions = self.model(**x)
        logits = predictions.logits
        
        if DEVICE == 'cpu':
            proba = torch.nn.functional.softmax(logits, dim = 1).detach().numpy()
        else:
            proba = torch.nn.functional.softmax(logits, dim = 1).to('cpu').detach().numpy()
        return proba
    
top5_df = pd.read_csv("/content/top5_wpopularitymeasures.csv")
batch_lyrics = top5_df['lyrics_w_puncta'].to_list()


sentiment_model_valence = SentimentModel('/content/sentiment_model_valence.pt')
batch_lyrics_probas_valence = sentiment_model_valence.batch_predict_proba(batch_lyrics)

predicted_class_labels_df_valence = pd.DataFrame(batch_lyrics_probas_valence, columns=['valence_0','valence_1'])
predicted_class_labels_df_valence.head()

top5_wlabels_valence = pd.merge(top5_df, predicted_class_labels_df_valence, how='inner', left_index=True, right_index=True)

sentiment_model_arousal = SentimentModel('/content/sentiment_model_arousal.pt')
batch_lyrics_probas_arousal = sentiment_model_arousal.batch_predict_proba(batch_lyrics)

predicted_class_labels_df_arousal = pd.DataFrame(batch_lyrics_probas_arousal, columns=['arousal_0','arousal_1'])
predicted_class_labels_df_arousal.head()


top5_wlabels_all = pd.merge(top5_wlabels_valence, predicted_class_labels_df_arousal, how='inner', left_index=True, right_index=True)

top5_wlabels_all.to_csv('/content/top5_wlabels_all.csv')






