import torch
import torch.nn as nn
import time
from tqdm import tqdm # for beautiful model training updates
import torch.nn.functional as F
from torch.optim import AdamW
from model import BertClassifier

def initialize_model(config):
    """Initialize the Bert Classifier, the optimizer and the learning rate scheduler.
    """
    # Instantiate Bert Classifier
    bert_classifier = BertClassifier(freeze_bert=False)

    # Tell PyTorch to run the model on GPU
    bert_classifier.to(config.device)

    # Create the optimizer
    optimizer = None
    if config.optimizer == 'AdamW':
        optimizer = AdamW(bert_classifier.parameters(),
                        lr=5e-5,    # Default learning rate
                        eps=1e-8    # Default epsilon value
                        )

    # Create the scheduler
    scheduler = None

    # Create the loss function
    loss_fn = None
    if config.criterion == 'BCE':
       loss_fn = nn.CrossEntropyLoss()

    return bert_classifier, optimizer, scheduler, loss_fn


def trainer(model,  device, trainloader, testloader, optimizer,epochs,criterion,scheduler = None, breaker = False):
  train_losses = [] # to capture train losses over training epochs
  train_accuracy = [] # to capture train accuracy over training epochs
  test_losses = [] # to capture test losses 
  test_accuracy = [] # to capture test accuracy 
  for epoch in range(epochs):
    print( f' EPOCH: {epoch+1} of {epochs}') 
    # train(model, device, trainloader, optimizer, epoch,criterion,train_accuracy,train_losses,scheduler) # Training Function
    train(model = model , 
          device = device, 
          train_loader = trainloader, 
          optimizer = optimizer, 
          epoch = epoch,
          criterion =criterion, # loss function
          train_accuracy = train_accuracy,
          train_losses = train_losses ,
          breaker = breaker,
          scheduler = None)
    print(f'training loop completed for epoch {epoch+1}')
    # test(model, device, testloader, criterion, test_accuracy, test_losses)   # Test Function
    test(model = model , 
          device = device, 
          test_loader = testloader, 
          criterion =criterion, # loss function
          test_accuracy = test_accuracy,
          test_losses = test_losses, 
          breaker = breaker)
    print(f'test loop completed for epoch {epoch+1}')

  return train_accuracy, train_losses, test_accuracy, test_losses


# # Training Function
def train(model, 
          device, 
          train_loader, 
          optimizer, 
          epoch,
          criterion, # loss function
          train_accuracy,
          train_losses,
          scheduler = None, 
          breaker = False):
    """ Train model
    """
    # Measure the elapsed time of each epoch
    t0_epoch, t0_batch = time.time(), time.time()

    
    pbar = tqdm(train_loader) # putting the iterator in pbar

    correct = 0 # for accuracy numerator
    processed =0 # for accuracy denominator

    model.train() # setting the model in training 
    
    for batch_idx, (input_ids, attn_masks, ratings) in enumerate(pbar):
        input_ids, attn_masks, ratings = input_ids.to(device), attn_masks.to(device), ratings.squeeze().to(device) #sending data to device
        
        optimizer.zero_grad() # setting gradients to zero to avoid accumulation

        logits = model(input_ids, attn_masks) # forward pass, result captured in logits (plural as there are many reviews in a batch)
        
        # logits_arg = logits.argmax(dim=1)
        # print(f'logits = {logits.shape} \n {logits} \n ratings = {ratings.shape} \n logit_args = {logits_arg.shape}')
        loss = criterion(logits,ratings) # capturing loss
        # print(f'loss = {loss.shape} \n {loss} ')


        train_losses.append(loss.item()) # to capture loss over many epochs

        loss.backward() # backpropagation
        
       
        # Clip the norm of the gradients to 1.0 to prevent "exploding gradients"
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0) 

        optimizer.step() # updating the params    
        
        # if scheduler:
        #   scheduler.step()

        if scheduler:
            if not isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step()
            else:
                scheduler.step(loss.item())
        
        
        preds = logits.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
        correct += preds.eq(ratings.view_as(preds)).sum().item()
        processed += len(input_ids)

        batch_time_elapsed = time.time() - t0_batch
        pbar.set_description(desc= f'Loss={loss.item()} Batch_id={batch_idx} Accuracy={100*correct/processed:0.2f} Time={batch_time_elapsed:0.3f}')

        train_accuracy.append(100*correct/processed) 
        if breaker and batch_idx > 1:
          break


    epoch_time_elapsed = time.time() - t0_epoch
    print(f'{epoch} epoch_time_elapsed = {epoch_time_elapsed:0.3f}')
    return

# # Test Function
def test(model, device, test_loader,criterion,test_accuracy,test_losses,breaker = False) :
  model.eval() # setting the model in evaluation mode
  test_loss = 0
  correct = 0 # for accuracy numerator

  with torch.no_grad():
    for batch_idx, (input_ids, attn_masks, ratings) in enumerate(test_loader):
      input_ids, attn_masks, ratings = input_ids.to(device), attn_masks.to(device), ratings.squeeze().to(device) #sending data to device

      outputs = model(input_ids, attn_masks) # forward pass, result captured in outputs logits (plural as there are many reviews in a batch)

      test_loss = criterion(outputs,ratings).item()  # sum up batch loss
      preds = outputs.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
      correct += preds.eq(ratings.view_as(preds)).sum().item()

      # print(test_loss)
      if breaker and batch_idx > 1:
          break

    test_loss /= len(test_loader.dataset) # average test loss
    test_losses.append(test_loss) # to capture loss over many batches

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
    test_loss, correct, len(test_loader.dataset),
    100. * correct / len(test_loader.dataset)))

    test_accuracy.append(100*correct/len(test_loader.dataset))

  return


def bert_predict(model, device, test_dataloader, breaker = False):
    """Perform a forward pass on the trained BERT model to predict probabilities
    on the test set.
    """
    # Put the model into the evaluation mode. The dropout layers are disabled during
    # the test time.
    model.eval()

    all_logits = []

    # For each batch in our test set...
    for batch_idx, batch in enumerate(test_dataloader):
        # Load batch to GPU
        b_input_ids, b_attn_mask = tuple(t.to(device) for t in batch)[:2]

        # Compute logits
        with torch.no_grad():
            logits = model(b_input_ids, b_attn_mask)
        all_logits.append(logits)

        if breaker and batch_idx > 1:
          break
    
    # Concatenate logits from each batch
    all_logits = torch.cat(all_logits, dim=0)

    # Apply softmax to calculate probabilities
    probs = F.softmax(all_logits, dim=1).cpu().numpy()

    return probs

# # # Training Function
# def train(model, 
#           device, 
#           train_loader, 
#           optimizer, 
#           epoch,
#           criterion, # loss function
#           train_accuracy,
#           train_losses,
#           scheduler = None):
#     """ Train model
#     """
#     # Measure the elapsed time of each epoch
#     t0_epoch, t0_batch = time.time(), time.time()

    
#     pbar = tqdm(train_loader) # putting the iterator in pbar

#     correct = 0 # for accuracy numerator
#     processed =0 # for accuracy denominator

#     model.train() # setting the model in training 

#     for batch_idx, (input_ids, attn_masks, ratings) in enumerate(pbar):
#         input_ids, attn_masks, ratings = input_ids.to(device), attn_masks.to(device), ratings.to(device) #sending data to device
        
#         optimizer.zero_grad() # setting gradients to zero to avoid accumulation

#         logits = model(input_ids, attn_masks) # forward pass, result captured in logits (plural as there are many reviews in a batch)
        
#         logits_arg = logits.argmax(dim=1)
#         print(f'logits = {logits.shape} \n ratings = {ratings.shape} \n logit_args = {logits_arg.shape}')
#         loss = criterion(logits_arg,ratings) # capturing loss

#         train_losses.append(loss.item()) # to capture loss over many epochs

#         loss.backward() # backpropagation
        
       
#         # Clip the norm of the gradients to 1.0 to prevent "exploding gradients"
#         torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0) 

#         optimizer.step() # updating the params    
        
#         # if scheduler:
#         #   scheduler.step()

#         if scheduler:
#             if not isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
#                 scheduler.step()
#             else:
#                 scheduler.step(loss.item())
        
        
#         preds = logits.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
#         correct += preds.eq(ratings.view_as(preds)).sum().item()
#         processed += len(input_ids)

#         batch_time_elapsed = time.time() - t0_batch
#         pbar.set_description(desc= f'Loss={loss.item()} Batch_id={batch_idx} Accuracy={100*correct/processed:0.2f} Time={batch_time_elapsed:0.3f}')

#         train_accuracy.append(100*correct/processed) 

#     epoch_time_elapsed = time.time() - t0_epoch
#     print(f'{epoch} epoch_time_elapsed = {epoch_time_elapsed:0.3f}')

# # # Test Function
# def test(model, device, test_loader,criterion,test_accuracy,test_losses) :
#   model.eval() # setting the model in evaluation mode
#   test_loss = 0
#   correct = 0 # for accuracy numerator

#   with torch.no_grad():
#     for (input_ids, attn_masks, ratings) in test_loader:
#       input_ids, ratings = input_ids.to(device), attn_masks.to(device), ratings.to(device) #sending data to device
#       outputs = model(input_ids, attn_masks) # forward pass, result captured in piputs logits (plural as there are many reviews in a batch)
#       # the outputs are in batch size x one hot vector 

#       test_loss = criterion(outputs,ratings).item()  # sum up batch loss
#       preds = outputs.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
#       correct += preds.eq(ratings.view_as(preds)).sum().item()

#     test_loss /= len(test_loader.dataset) # average test loss
#     test_losses.append(test_loss) # to capture loss over many batches

#     print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
#     test_loss, correct, len(test_loader.dataset),
#     100. * correct / len(test_loader.dataset)))

#     test_accuracy.append(100*correct/len(test_loader.dataset))