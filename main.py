from transformers import BertTokenizer
from torch.utils.data import  DataLoader,RandomSampler
import torch.nn as nn
from transformers import logging
logging.set_verbosity_error()

from data import *
from utils import *
from preprocess import *
from model import BertClassifier
from train import *
from config import *

if __name__ == "__main__":
    # getting configurations
    data_config = DataConfig()
    experiment_config = ExperimentConfig()

    # instantiate the BERT tokenizer and model
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    # dataset creation - train and validation
    ds_books_train, ds_books_test = get_dataset(domain ='B', 
                                          tokenizer=tokenizer,
                                          config = data_config,
                                          transform=True,
                                          labeled = True, 
                                          split = True, 
                                          test_size = 0.2,
                                          only_df= False)
    # dataloader creation - train and validation
    dl_books_train = DataLoader(ds_books_train, 
                                sampler=RandomSampler(ds_books_train), # randomization in train dl only
                                batch_size=experiment_config.batch_size)
    dl_books_test = DataLoader(ds_books_test,  
                                batch_size=experiment_config.batch_size,
                                shuffle=False )

    if experiment_config.mode == 'debug':
        batch = next(iter(dl_books_test))
        print(f'Sample Batch :\n {batch}')

    # Model training preparation
    model, optimizer, scheduler, loss_fn = initialize_model(config = experiment_config)
    
    if experiment_config.mode == 'debug':
        device = experiment_config.device
        print(f'Model :\n {model}')
        print(f'Device: {device}')
        print(f'One forward pass of model..in progress \n')
        input_ids, attention_masks, ratings = next(iter(dl_books_test))
        print(f'Sample Batch :\n {input_ids.shape, attention_masks.shape, ratings.shape }')
        outputs = model(input_ids.to(device), attention_masks.to(device))
        print(f'outputs :\n {outputs}')

    # Training and Validating the model
    train_accuracy, train_losses, test_accuracy, test_losses = trainer(model = model,  
                                                                        device = experiment_config.device, 
                                                                        trainloader = dl_books_train, 
                                                                        testloader= dl_books_test, 
                                                                        optimizer = optimizer,
                                                                        epochs = experiment_config.num_epochs,
                                                                        criterion =loss_fn,
                                                                        scheduler = scheduler, 
                                                                        breaker = False if (experiment_config.device =='cuda') else True)

    print (f'train_accuracy = {train_accuracy} ') 
    print (f'train_losses = {train_losses}')
    print (f'test_accuracy = {test_accuracy}')
    print (f'test_losses = {test_losses}')
    
    # only if we want to run test with unseen data
    test_dataloader = dl_books_test
    if experiment_config.test_unseen:
        # Compute predicted probabilities on the test /unseen set
        # Dataset for test / unseen set
        ds_books_unlabeled = get_dataset(domain ='B', 
                                        tokenizer=tokenizer,
                                        config = data_config,
                                        transform=True,
                                        labeled = False, 
                                        split = False, 
                                        only_df= False)
        # Loader for test / unseen set
        dl_books_unlabeled = DataLoader(ds_books_unlabeled, 
                                    batch_size=experiment_config.batch_size, 
                                    shuffle=False )
        
        test_dataloader = dl_books_unlabeled

    if experiment_config.mode == 'debug':
        batch = next(iter(test_dataloader))
        print(f'Sample Batch for test loader data :\n {batch}')

    # Predicting after passing through the model
    probs = bert_predict(model = model, 
                        device = experiment_config.device, 
                        test_dataloader = test_dataloader,
                        breaker = False if (experiment_config.device =='cuda') else True)
    
    if experiment_config.mode == 'debug':
        print(f'After passing through the model, len of output:\n {len(probs)}')

    # Acquiring information from ground truth
    y_true = []
    for batch in test_dataloader:
        for element in batch[2]: # ratings appears at index = 2 
            y_true.append(element)

    y_true = np.array(y_true[:len(probs)]) # conversion to np.ndarray, only take the top values similar to probs

    evaluate_confusion_matrix(probs, y_true)
    evaluate_roc(probs, y_true)