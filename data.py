from dataclasses import dataclass
import torch
from torch.utils.data import Dataset
import os
from sklearn.model_selection import train_test_split
import numpy as np

from data import *
from utils import *
from preprocess import *

# @dataclass
# class DataConfig:
#     base_path = "D:\Peeyush\Phd\domain-adaptation"
#     data_path = os.path.join(base_path,"dataset","amazon-multi-domain-sentiment-dataset")
#     model_path = os.path.join(base_path,"saved_model")
#     B = 'books'
#     D = 'dvd'
#     E = 'electronics'
#     K = 'kitchen_&_housewares'
#     dict_domains = {'B' : 'books',
#                     'D' : 'dvd',
#                     'E' : 'electronics',
#                     'K' : 'kitchen_&_housewares'}
#     labeled_files = ['positive','negative']
#     unlabeled_files = ['unlabeled']
#     ratings_mapping = {'1.0':0,'2.0':0,'3.0':0, '4.0':1, '5.0':1}
#     max_length = 256
#     batch_size = 2
#     num_worker =0



# config = DataConfig() # Global variable

def preprocess_text(text):
    # convert to lowercase
    text = text.lower()
    # remove leading/trailing whitespace
    text = text.strip()
    return text

def get_dataset(domain, tokenizer, config, transform = None, labeled = True, split = True, test_size = 0.2, convert_ratings_into_label = True, only_df = False):
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
        if only_df:
            return train_df, test_df 
        return AmazonDataset(train_df,tokenizer = tokenizer), AmazonDataset(test_df,tokenizer = tokenizer)
        
    else:
        if only_df:
            return df
        return AmazonDataset(df, tokenizer = tokenizer)

class AmazonDataset(Dataset):
    def __init__(self, df, tokenizer, max_len = 512):
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.reviews = df['review'].to_numpy()
        self.ratings = df['rating'].to_numpy()

    def __len__(self):
        return len(self.reviews)

    def __getitem__(self, index):
        review = str(self.reviews[index])
        rating = self.ratings[index]

        encoding = self.tokenizer.encode_plus(
            preprocess_text(review),
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_token_type_ids=False,
            return_attention_mask=True,
            return_tensors='pt',
        )

        input_ids = encoding['input_ids'][0]
        attention_mask = encoding['attention_mask'][0]

        return (input_ids.to(torch.long), #torch.tensor(input_ids, dtype=torch.long),
            attention_mask.to(torch.long), #torch.tensor(attention_mask, dtype=torch.long),
            torch.tensor(rating, dtype=torch.long)
       )



# def encode(text, tokenizer):
#     return tokenizer(text,
#                      add_special_tokens=True,
#                      return_attention_mask=True,
#                      return_token_type_ids = False,
#                      padding = 'max_length',
#                      max_length =512, 
#                      truncation = True,
#                      return_tensors ='pt')

# class AmazonReviewsDataset(Dataset):
#     def __init__(self, df, tokenizer, transform = None):
#         if transform:
#             self.reviews = [encode(preprocess_text(review),tokenizer=tokenizer) 
#                             for review in df['review']]
#         else:
#             self.reviews = [encode(review,tokenizer=tokenizer) 
#                             for review in df['review']]
        
#         self.ratings = [rating for rating in df['rating']]
    
#     def classes(self):
#         return self.ratings
    
#     def __len__(self):
#         return len(self.ratings)
    
#     def get_batch_ratings(self, idx):
#         # Fetch a batch of ratings
#         return np.array(self.ratings[idx])

#     def get_batch_reviews(self, idx):
#         # Fetch a batch of inputs
#         return self.reviews[idx]

#     def __getitem__(self, idx):

#         batch_reviews = self.get_batch_reviews(idx)
#         batch_ratings = self.get_batch_ratings(idx)

#         return batch_reviews, batch_ratings

# class AmazonReviewsDataset(Dataset):
#     def __init__(self, df, tokenizer, transform = None):
#         self.df = df
#         self.tokenizer = tokenizer
#         self.transform = transform

#     def __len__(self):
#         return len(self.df)

#     def __getitem__(self, idx):
#         review = self.df.iloc[idx]['review']
#         rating = self.df.iloc[idx]['rating']

#         # apply data cleaning transform if defined
#         print('\nOriginal: \n', review)
#         if self.transform:
#             # transform_fn = self.transform
#             # review, rating= transform_fn(review, rating)
#             review = preprocess_text(review)

#         # tokenize and encode text data
#         print("\nProcessed: \n",review)
#         # inputs = self.tokenizer.encode_plus(
#         #     review,
#         #     add_special_tokens=True,
#         #     max_length=512,
#         #     padding='max_length',
#         #     is_split_into_words=True,
#         #     truncation=True,
#         #     return_attention_mask=False,
#         #     return_tensors='pt',
#         #     return_token_type_ids = False
#         # ) 
#         # print(review)
#         inputs = self.tokenizer.batch_encode_plus(
#             # review.tolist(),
#             review,
#             add_special_tokens=True,
#             max_length=512,
#             padding='max_length',
#             is_split_into_words=True,
#             truncation=True,
#             return_attention_mask=False,
#             return_tensors='pt',
#             return_token_type_ids = False
#         ) 
         
#         return inputs, rating

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
