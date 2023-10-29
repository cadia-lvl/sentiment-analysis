


import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from pprint import pprint
from transformers import AutoTokenizer, AutoModelForSequenceClassification

class SentimentDataset(Dataset):
    # Constructor Function 
    def __init__(self, reviews, targets, tokenizer, max_len):
        self.reviews = reviews
        self.targets = targets
        self.tokenizer = tokenizer
        self.max_len = max_len
    
    # Length magic method
    def __len__(self):
        return len(self.reviews)
    
    # get item magic method
    def __getitem__(self, item):
        review = str(self.reviews[item])
        target = self.targets[item]
        
        # Encoded format to be returned 
        encoding = self.tokenizer.encode_plus(
            review,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            pad_to_max_length=True,
            return_attention_mask=True,
            return_tensors='pt',
        )
        
        return {
            'review_text': review,
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'targets': torch.tensor(target, dtype=torch.long)
        }
    
class SentimentData(Dataset):
    def __init__(self, reviews, sentiments, tokenizer, max_len):
        self.tokenizer = tokenizer
        self.text = reviews
        self.targets = sentiments
        self.max_len = max_len

    def __len__(self):
        return len(self.text)

    def __getitem__(self, index):
        text = str(self.text[index])
        text = " ".join(text.split())
        
        inputs = self.tokenizer.encode_plus(
            text,
            None,
            add_special_tokens=True,
            max_length=self.max_len,
            pad_to_max_length=True,
            return_token_type_ids=True
        )
        ids = inputs['input_ids']
        mask = inputs['attention_mask']
        token_type_ids = inputs["token_type_ids"]


        return {
            'ids': torch.tensor(ids, dtype=torch.long),
            'mask': torch.tensor(mask, dtype=torch.long),
            'token_type_ids': torch.tensor(token_type_ids, dtype=torch.long),
            'targets': torch.tensor(self.targets[index], dtype=torch.float)
        }



class RoBERTaClassificationReport:
    def __init__(self, model, tokenizer, review, sentiment, device):
        self.model = model
        self.tokenizer = tokenizer
        self.test_data = self.create_data_loader(review, sentiment, tokenizer)
        self.device = device

    def create_data_loader(self, review, sentiment, tokenizer, max_len=512, batch_size=8) -> DataLoader:
        ds = SentimentDataset(
            reviews=review.to_numpy(),
            targets=sentiment.to_numpy(),
            tokenizer=tokenizer,
            max_len=max_len
        )
    
        return DataLoader(
            ds,
            batch_size=batch_size,
            num_workers=0
        )
    
    # 13-15:30 miðv (8)
    # 13-14,14:30-15:30 fim (9)
    

    def generate_report(self) -> (str | dict):
        self.model.eval()
        y_true = []
        y_pred = []
        with torch.no_grad():
            for batch in self.test_data:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['targets'].to(self.device)
                outputs = self.model(input_ids, attention_mask=attention_mask, return_dict=True)
                prediction = torch.max(outputs.logits, dim=1)
                y_true.extend(labels.tolist())
                y_pred.extend(prediction.indices.tolist())
        report = classification_report(y_true, y_pred, output_dict=True)
        return report


class DataFrameLoader():
    def __init__(self, pdf_src, sample_size=None, random_state=42):
        
        self.df = pd.read_csv(pdf_src)
        if "Unnamed: 0" in self.df.columns:
            self.df.drop(['Unnamed: 0'], axis=1, inplace=True)
        self.df['sentiment'] = self.df.sentiment.apply(lambda sentiment: 1 if sentiment == 'positive' else 0)
        
        if sample_size is not None:
            self.df = self.df.sample(n=sample_size, random_state=random_state)
        
        # 70% train, 15% validation, 15% test
        X_train, X_temp, y_train, y_temp = train_test_split(self.df["review"], self.df["sentiment"], test_size=0.3, random_state=random_state)
        X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=random_state)
        
        self.X_train = X_train
        self.y_train = y_train

        self.X_val = X_val
        self.y_val = y_val

        self.X_test = X_test
        self.y_test = y_test


if __name__ == '__main__':
    import gc
    gc.collect()
    torch.cuda.empty_cache()
    device = "cuda"
    data = [
        {
            'folder': './roberta-batch8-unprocessed_model/',
            'filename': '../IMDB-Dataset.csv'
        },
        {
            'folder': './roberta-batch8-unprocessed_model/',
            'filename': '../IMDB-Dataset-Processed.csv'
        }
    ]
    for d in data:
        folder = d['folder']
        filename = d['filename']
        print("Loading model from folder {} using file {}".format(folder, filename))
        dfl = DataFrameLoader(filename)
        model = AutoModelForSequenceClassification.from_pretrained(folder)
        model.to(device)
        tokenizer = AutoTokenizer.from_pretrained(folder)
        report = RoBERTaClassificationReport(model, tokenizer, dfl.X_test, dfl.y_test, device)
        pprint(report.generate_report())
