from dataclasses import dataclass
import torch
from torch.utils.data import Dataset
import os
from sklearn.model_selection import train_test_split

from data import *
from utils import *
from preprocess import *

@dataclass
class DataConfig:
    base_path = "D:\Peeyush\Phd\domain-adaptation"
    data_path = os.path.join(base_path,"dataset","amazon-multi-domain-sentiment-dataset")
    B = 'books'
    D = 'dvd'
    E = 'electronics'
    K = 'kitchen_&_housewares'
    dict_domains = {'B' : 'books',
                    'D' : 'dvd',
                    'E' : 'electronics',
                    'K' : 'kitchen_&_housewares'}
    labeled_files = ['positive','negative']
    unlabeled_files = ['unlabeled']
    ratings_mapping = {'1.0':0,'2.0':0,'3.0':0, '4.0':1, '5.0':1}

config = DataConfig() # Global variable

# def preprocess

def get_dataset(domain, tokenizer, transform = None, labeled = True, split = True, test_size = 0.2, convert_ratings_into_label = True):
    domain_folder = os.path.join(config.data_path,config.dict_domains[domain])
    list_file_paths = get_files(domain_folder)
    df = pd.DataFrame()
    for file_path in list_file_paths:
        file_name = get_file_name(file_path)
        if labeled:
            for labeled_file in config.labeled_files:
                if labeled_file in file_name:
                    data_df = get_data_df(file_path=file_path)
                    df = pd.concat([df,data_df], ignore_index=True)
        else:
            for unlabeled_file in config.unlabeled_files:
                if unlabeled_file in file_name:
                    data_df = get_data_df(file_path=file_path)
                    df = pd.concat([df,data_df], ignore_index=True)
    
    if convert_ratings_into_label:
        df = convert_ratings_into_labels(df, mapping = config.ratings_mapping)
    if split:
        train_df, test_df = train_test_split(df,  test_size=test_size, random_state=42)
        return AmazonReviewsDataset(train_df,tokenizer = tokenizer, transform = transform), AmazonReviewsDataset(test_df,tokenizer = tokenizer, transform = transform)
    else:
        return AmazonReviewsDataset(df, tokenizer = tokenizer, transform = transform)

class AmazonReviewsDataset(Dataset):
    def __init__(self, df, tokenizer, transform = None):
        self.df = df
        self.tokenizer = tokenizer
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        review = self.df.iloc[idx]['review']
        rating = self.df.iloc[idx]['rating']

        # apply data cleaning transform if defined
        if self.transform:
            transform_fn = self.transform
            review, rating= transform_fn(review, rating)

        # tokenize and encode text data
        # print("------------",review, type(review))
        # inputs = self.tokenizer.encode_plus(
        #     review,
        #     add_special_tokens=True,
        #     max_length=512,
        #     padding='max_length',
        #     is_split_into_words=True,
        #     truncation=True,
        #     return_attention_mask=False,
        #     return_tensors='pt',
        #     return_token_type_ids = False
        # ) 
        inputs = self.tokenizer.batch_encode_plus(
            review.tolist(),
            add_special_tokens=True,
            max_length=512,
            padding='max_length',
            is_split_into_words=True,
            truncation=True,
            return_attention_mask=False,
            return_tensors='pt',
            return_token_type_ids = False
        ) 
         
        return inputs, rating

# if __name__ == "__main__":
#     pass

    # books_train, books_test = get_dataset(domain ='B', labeled = True, split = True, test_size = 0.2)
    # # books_unlabled = get_dataset(domain = 'B', labeled= False, split= False)
    # # print(len(books_train),len(books_test), len(books_unlabled))
    # print(books_train[0])


    
    # books_dataset = AmazonReviewsDataset("B",labeled=False)

    # domain_dataset ={}
    
    # for domain in config.dict_domains.keys():
    #     print("----------",domain,"-----------")
    #     domain_dataset['labeled'] = AmazonReviewsDataset(str(domain), labeled= True)
    #     domain_dataset['unlabeled'] = AmazonReviewsDataset (str(domain), labeled= False)
    #     print(domain_dataset['labeled'].df.head())
    #     print('-'*20)
    #     print(domain_dataset['labeled'].df.tail())
    #     print('*'*20)
    #     print(domain_dataset['unlabeled'] .df.tail())

        
    # print(books_dataset.df.head())
    # print('-'*20)
    # print(books_dataset.df.tail())
    # t = 0
    # print(books_dataset.tempdf['review'][t])

    # print(Kitchen_folder.domain_folder)
