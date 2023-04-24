import torch
import torchvision.transforms as transforms

# data_transform = transforms.Lambda(lambda x: clean_text_and_label(x[0], x[1]))

class Transform:
    def __init__(self):
        pass
    
    def __call__(self, text, label):
        text = text.apply(preprocess_text)
        # convert label to long tensor
        label = torch.tensor(label.values, dtype=torch.long)
        return text, label

    
def preprocess_text(text):
    # convert to lowercase
    text = text.lower()
    # remove leading/trailing whitespace
    text = text.strip()
    return text

def convert_ratings_into_labels(df, 
                                mapping = {'1.0':0,'2.0':0,'3.0':0, '4.0':1, '5.0':1}):
     df['rating'] = df['rating'].map(mapping) 
     return df