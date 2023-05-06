import torch
import torch.nn as nn
from transformers import BertModel, logging
logging.set_verbosity_error()



class BertClassifier(nn.Module):
    """Bert Model for Classification Tasks.
    """
    def __init__(self, freeze_bert=False, hidden = 100, dropout=0.2):
        """
        @param    bert: a BertModel object
        @param    classifier: a torch.nn.Module classifier
        @param    freeze_bert (bool): Set `False` to fine-tune the BERT model
        @param    dropout: value of dropout to be applied
        """
        super(BertClassifier, self).__init__()

        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.classifier = nn.Sequential(nn.Linear(768,hidden),
                                        nn.ReLU(),
                                        nn.Dropout(dropout),
                                        nn.Linear(hidden,2)
                                        )
        # Freeze the BERT model
        if freeze_bert:
            for param in self.bert.parameters():
                param.requires_grad = False
        
    
    def forward(self, input_ids, attention_mask):
        """
        Feed input to BERT and the classifier to compute logits.
        @param    input_ids (torch.Tensor): an input tensor with shape (batch_size,
                      max_length)
        @param    attention_mask (torch.Tensor): a tensor that hold attention mask
                      information with shape (batch_size, max_length)
        @return   logits (torch.Tensor): an output tensor with shape (batch_size,
                      num_labels)
        """
        # Feed input to BERT
        outputs = self.bert(input_ids=input_ids,
                            attention_mask=attention_mask)
        
        # Extract the last hidden state of the token `[CLS]` for classification task
        # outputs[0] is a tensor of shape (batch_size, sequence_length, hidden_size)
        # [:, 0, :] selects all the examples in a batch (first :), 0 means CLS- fist token,
        # last : means all the hidden size

        last_hidden_state_cls = outputs[0][:, 0, :] 

        # Feed input to classifier to compute logits
        logits = self.classifier(last_hidden_state_cls)

        return logits


# if __name__ == "__main__":
#     from transformers import BertTokenizer
#     # Load the pre-trained model and tokenizer
#     tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
#     model = BertModel.from_pretrained('bert-base-uncased')


#     # Tokenize input text
#     text = "This is life"
#     tokens = tokenizer.encode_plus(text, add_special_tokens=True, return_tensors='pt')
#     print(tokens)

#     # Pass input through the model to get hidden states
#     outputs = model(input_ids=tokens['input_ids'], attention_mask=tokens['attention_mask'])
#     hidden_states = outputs[0][:, 0, :]
   

#     # Print the shape of the hidden states
#     print(hidden_states)
#     print(hidden_states.shape)