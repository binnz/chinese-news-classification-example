import torch
import tensorflow as tf
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from pytorch_pretrained_bert import BertTokenizer, BertConfig
from pytorch_pretrained_bert import BertAdam, BertForSequenceClassification
from tqdm import tqdm, trange
import pandas as pd
import io
import numpy as np
import matplotlib.pyplot as plt
import os
import argparse

parser = argparse.ArgumentParser()

## Required parameters
parser.add_argument(
    "-- ", default=None, type=str, required=True, help="The input data_dir")
parser.add_argument(
    "--output_dir",
    default=None,
    type=str,
    required=True,
    help="The output directory where the model checkpoints saved")
parser.add_argument("--max_seq_length", default=64, type=int, help="")
parser.add_argument(
    "--batch_size",
    default=16,
    type=int,
    help="Total batch size for training and eval.")
parser.add_argument(
    "--use_gpu",
    default=True,
    type=str,
    help="Whether not to use CUDA when available")
parser.add_argument(
    "--is_add_key_words",
    default=True,
    type=str,
    help="Whether or not consider key words in dataset special for this dataset"
)
parser.add_argument(
    "--epochs",
    default=4,
    type=int,
    help="Total number of training epochs to perform.")
args = parser.parse_args()
# check gpu work well or use cpu

device_name = tf.test.gpu_device_name()
print(device_name)
use_gpu = False
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if device == "cuda":
    use_gpu = True
    n_gpu = torch.cuda.device_count()
    print("gpu number:", n_gpu, torch.cuda.get_device_name(0))


def dropcomma(sent):
    if isinstance(sent, str):
        return sent.replace(',', '')
    else:
        return ''


def add_keywords(x):
    if args.is_add_key_words:
        return np.array([dropcomma(xi) for xi in x])
    else:
        return np.full_like(x, '')


# prepare data
df = pd.read_csv(
    os.path.join(
        os.path.expanduser("~"), "ml-project/muse/toutiao_cat_data.txt"),
    delimiter='_!_',
    header=None)
print("Data load success")
sentences = df[3].values
key_words = df[4].values
labels = df[1].values
labels[labels < 105] -= 100
labels[labels > 105] -= 101
print("Data number", sentences.shape)
print("Tokenize start...")

# Here we can test whether after add key_words to tokens, the system perfrom better by set 'is_add_key_words' param
key_words_drop_comma = add_keywords(key_words)

# We need to add special tokens at the beginning and end of each sentence for BERT to work properly
sentences = [
    "[CLS] " + sentence + keyword + " [SEP]" for sentence in sentences
    for keyword in key_words_drop_comma if sentence is not None
]
print("sentence prepare ok")
tokenizer = BertTokenizer.from_pretrained(
    'bert-base-chinese', do_lower_case=True)
tokenized_texts = [tokenizer.tokenize(sent) for sent in sentences]
print("Tokenize success")
MAX_LEN = args.max_seq_length
input_ids = pad_sequences(
    [tokenizer.convert_tokens_to_ids(txt) for txt in tokenized_texts],
    maxlen=MAX_LEN,
    dtype="long",
    truncating="post",
    padding="post")

# Create attention masks
attention_masks = []

# Create a mask of 1s for each token followed by 0s for padding
for seq in input_ids:
    seq_mask = [float(i > 0) for i in seq]
    attention_masks.append(seq_mask)

# Use train_test_split to split our data into train and validation sets for training

train_inputs, validation_inputs, train_labels, validation_labels = train_test_split(
    input_ids, labels, random_state=1, test_size=0.2)
train_masks, validation_masks, _, _ = train_test_split(
    attention_masks, input_ids, random_state=1, test_size=0.2)

batch_size = args.batch_size
train_inputs = torch.tensor(train_inputs)
validation_inputs = torch.tensor(validation_inputs)
train_labels = torch.tensor(train_labels)
validation_labels = torch.tensor(validation_labels)
train_masks = torch.tensor(train_masks)
validation_masks = torch.tensor(validation_masks)

# Create an iterator of our data with torch DataLoader. This helps save on memory during training because, unlike a for loop,
# with an iterator the entire dataset does not need to be loaded into memory

train_data = TensorDataset(train_inputs, train_masks, train_labels)
train_sampler = RandomSampler(train_data)
train_dataloader = DataLoader(
    train_data, sampler=train_sampler, batch_size=batch_size)

validation_data = TensorDataset(validation_inputs, validation_masks,
                                validation_labels)
validation_sampler = SequentialSampler(validation_data)
validation_dataloader = DataLoader(
    validation_data, sampler=validation_sampler, batch_size=batch_size)
print("Data prepare Finished")
model = BertForSequenceClassification.from_pretrained(
    "bert-base-chinese", num_labels=16)
if use_gpu:
    model.cuda()

param_optimizer = list(model.named_parameters())
no_decay = ['bias', 'gamma', 'beta']
optimizer_grouped_parameters = [
    {
        'params':
        [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
        'weight_decay_rate':
        0.01
    },
    {
        'params':
        [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
        'weight_decay_rate':
        0.0
    }
]
optimizer = BertAdam(optimizer_grouped_parameters, lr=2e-5, warmup=.1)


# Function to calculate the accuracy of our predictions vs labels
def flat_accuracy(preds, labels):
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)


# Store our loss and accuracy for plotting
train_loss_set = []

# Number of training epochs (authors recommend between 2 and 4)
epochs = 4
print("Starting training")
# trange is a tqdm wrapper around the normal python range
for epoch in trange(epochs, desc="Epoch"):

    # Training

    # Set our model to training mode (as opposed to evaluation mode)
    model.train()

    # Tracking variables
    tr_loss = 0
    nb_tr_examples, nb_tr_steps = 0, 0

    # Train the data for one epoch
    for step, batch in enumerate(train_dataloader):
        # Add batch to GPU
        print("Epoch:", epoch, "step:", step)
        # batch = tuple(t.to(device) for t in batch)
        # Unpack the inputs from our dataloader
        b_input_ids, b_input_mask, b_labels = batch
        # Clear out the gradients (by default they accumulate)
        optimizer.zero_grad()
        # Forward pass
        b_input_ids.to(device)
        b_input_mask.to(device)
        b_labels.to(device)
        loss = model(
            b_input_ids,
            token_type_ids=None,
            attention_mask=b_input_mask,
            labels=b_labels)
        train_loss_set.append(loss.item())
        # Backward pass
        loss.backward()
        # Update parameters and take a step using the computed gradient
        optimizer.step()

        # Update tracking variables
        tr_loss += loss.item()
        nb_tr_examples += b_input_ids.size(0)
        nb_tr_steps += 1

    print("Train loss: {}".format(tr_loss / nb_tr_steps))

    # Validation

    # Put model in evaluation mode to evaluate loss on the validation set
    model.eval()

    # Tracking variables
    eval_loss, eval_accuracy = 0, 0
    nb_eval_steps, nb_eval_examples = 0, 0

    # Evaluate data for one epoch
    for batch in validation_dataloader:
        # Add batch to GPU
        #batch = tuple(t.to(device) for t in batch)
        # Unpack the inputs from our dataloader
        b_input_ids, b_input_mask, b_labels = batch
        # Telling the model not to compute or store gradients, saving memory and speeding up validation
        with torch.no_grad():
            # Forward pass, calculate logit predictions
            b_input_ids.to(device)
            b_input_mask.to(device)
            b_labels.to(device)
            logits = model(
                b_input_ids, token_type_ids=None, attention_mask=b_input_mask)

        # Move logits and labels to CPU
        logits = logits.detach().cpu().numpy()
        label_ids = b_labels.to('cpu').numpy()

        tmp_eval_accuracy = flat_accuracy(logits, label_ids)

        eval_accuracy += tmp_eval_accuracy
        nb_eval_steps += 1

    print("Validation Accuracy: {}".format(eval_accuracy / nb_eval_steps))

torch.save(model, args.output_dir)