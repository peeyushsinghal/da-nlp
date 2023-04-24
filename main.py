from transformers import BertTokenizer
# from transformers import BertForSequenceClassification

from torch.utils.data import  DataLoader

from data import *
from utils import *
from preprocess import *


if __name__ == "__main__":
    # instantiate the BERT tokenizer and model
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    # model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)

    # define the data transform to be applied to each sample
    data_transform = Transform() # as part of preprocess

    books_train, books_test = get_dataset(domain ='B', 
                                          tokenizer=tokenizer,
                                          transform=data_transform,
                                          labeled = True, split = True, test_size = 0.2)
    
    books_unlabled = get_dataset(domain ='B', 
                                          tokenizer=tokenizer,
                                          transform=data_transform,
                                          labeled = False, split = False)
    
    a = books_unlabled[1:3]
    # print(a[0]['input_ids'].shape)
    print(a)

    # books_train_dl = DataLoader(books_train, batch_size=32, num_workers=4, shuffle=True)

    # print(next(iter(books_train_dl)))