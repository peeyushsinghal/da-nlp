import os
from dataclasses import dataclass
import torch

BASE_PATH = "D:\Peeyush\Phd\domain-adaptation"
@dataclass
class DataConfig:
    data_path = os.path.join(BASE_PATH,"dataset","amazon-multi-domain-sentiment-dataset")
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
    max_length = 256


@dataclass
class ExperimentConfig:
    model_path = os.path.join(BASE_PATH,"saved_model")
    # machine = 'cpu'
    mode = 'debug'
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device == "cuda":
        batch_size = 32
    else:
        batch_size = 2
    optimizer = 'AdamW'
    scheduler = None
    criterion = 'BCE'
    test_unseen = False
    num_worker =0
    num_epochs = 2

