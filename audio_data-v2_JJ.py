#!/usr/bin/env python
# coding: utf-8

# # Paperspace Gradient: PyTorch Quick Start
# Last modified: Sep 27th 2022

# ## Purpose and intended audience
# 
# This Quick Start tutorial demonstrates PyTorch usage in a Gradient Notebook. It is aimed at users who are relatviely new to PyTorch, although you will need to be familiar with Python to understand PyTorch code.
# 
# We use PyTorch to
# 
# - Build a neural network that classifies FashionMNIST images
# - Train and evaluate the network
# - Save the model
# - Perform predictions
# 
# followed by some next steps that you can take to proceed with using Gradient.
# 
# The material is based on the original [PyTorch Quick Start](https://pytorch.org/tutorials/beginner/basics/quickstart_tutorial.html) at the time of writing this notebook.
# 
# See the end of the notebook for the original copyright notice.

# ## Check that you are on a GPU machine
# 
# The notebook is designed to run on a Gradient GPU machine (as opposed to a CPU-only machine). The machine type, e.g., A4000, can be seen by clicking on the Machine icon on the left-hand navigation bar in the Gradient Notebook interface. It will say if it is CPU or GPU.
# 
# ![quick_start_pytorch_images/example_instance_type.png](quick_start_pytorch_images/example_instance_type.png)
# 
# The *Creating models* section below also determines whether or not a GPU is available for us to use.
# 
# If the machine type is CPU, you can change it by clicking *Stop Machine*, then the machine type displayed to get a drop-down list. Select a GPU machine and start up the Notebook again.
# 
# For help with machines, see the Gradient documentation on [machine types](https://docs.paperspace.com/gradient/machines/) or [starting a Gradient Notebook](https://docs.paperspace.com/gradient/explore-train-deploy/notebooks).

# ## Working with data
# 
# PyTorch has two [primitives to work with data](https://pytorch.org/docs/stable/data.html):
# ``torch.utils.data.DataLoader`` and ``torch.utils.data.Dataset``.
# ``Dataset`` stores the samples and their corresponding labels, and ``DataLoader`` wraps an iterable around
# the ``Dataset``.

# In[1]:


import torch
from torch import nn
from torch.utils.data import DataLoader

import os

from torch.utils.data import Dataset
import pandas as pd
import torchaudio


# In[2]:


# https://stackoverflow.com/posts/8384788/revisions
#
import ntpath
def path_leaf(path):
    head, tail = ntpath.split(path)
    return tail or ntpath.basename(head)
#
def list_files_in_path(path):
    import pathlib
    myFileList = []
    mypath = pathlib.Path(path)

    # Using a for loop
    for item in mypath.rglob("*"):
        if item.is_file():
            myFileList.append(item)

    return myFileList
#    


# In[3]:


class SoundDataset(Dataset):

    def __init__(self, SR = 0, audio_dir="./data",list_of_files=[]):
        import glob
        #self.annotations = pd.read_csv(annotations_file)
        self._audio_dir = audio_dir
        print("init:{}".format(self._audio_dir))
        self.list_of_files = list_files_in_path(self._audio_dir)
        print(self.list_of_files[0])

    #"""
    def _list_of_files(self):
        return list(self.list_of_files)
    #"""
        
    def __len__(self):
        #https://stackoverflow.com/questions/72274073/python-count-files-in-a-directory-and-all-its-subdirectories
        #sum(1 for _, _, files in os.walk(self.audio_dir) for f in files)
        return len(self.list_of_files)

    def __getitem__(self, index):
        from pathlib import Path
        import librosa
        audio_sample_path = self.list_of_files[index]
        label = Path(path_leaf(audio_sample_path)).stem
        #signal, sr = librosa.load(audio_sample_path)
        signal,sr=librosa.load(audio_sample_path)
        self.SR = sr
        return signal, label    

    """
    def _get_audio_sample_path(self, index):
        # the name of the file is the label to the data
        #
        path = os.path.join(self.audio_dir, fold, self.annotations.iloc[
            index, 0])
        return path
    
    def _get_audio_sample_label(self, index):
        return self.annotations.iloc[index, 6]
    """


# In[4]:


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import os
    import subprocess
    #
    AUDIO_DIR = "./data"
    usd = SoundDataset(AUDIO_DIR)
    print(f"There are {len(usd)} samples in the dataset.")
    signal, label = usd[0]
    plt.plot(signal)
    plt.show()
    #


# In[ ]:


pip install librosa


# In[ ]:


"""
# DONE THIS ALREADY 
#
# ref: https://stackoverflow.com/a/64908307
#
from pathlib import Path, PureWindowsPath
all_ogg_files = list_files_in_path(r'./data')
print(PureWindowsPath(all_ogg_files[0]).as_posix())
try:
    os.mkdir('./data_mp3')
except FileExistsError:
    pass
for ogg in all_ogg_files:
    ogg_name = str(PureWindowsPath(ogg).as_posix()) #https://stackoverflow.com/a/67536440
    temp_name = ogg_name.split(".")[0]+".mp3"
    mp3_name = "data_mp3/{}".format(temp_name[5:])
    print("ogg = {} mp3 = {}".format(str(ogg_name),str(mp3_name)))
    subprocess.run(["ffmpeg", "-i", ogg_name, mp3_name])
"""


# In[ ]:


pip install openai


# In[ ]:


import os
import openai
    # Note: you need to be using OpenAI Python v0.27.0 for the code below to work

API_KEY='enter API key'

os.environ['OPENAI_Key']=API_KEY
openai.api_key=os.environ['OPENAI_Key']


# In[ ]:


pip install num2words


# ## Text Similarity to compare output of Whisper with Label

# In[ ]:


from num2words import num2words
import re

def num_units(s):
    pattern = re.compile(r"\d+(?:[.,-x]{0,}\d*?\s?[a-zA-Z°µ ][a-zA-Z0-9°µ/*^-]*)",re.I)
    return(re.findall(pattern,s))
#

def isolate_num_units(textstr,num_units_to_words=None):
    """
    Find the number+units in a given string (as isolated by "num_units" method)
    Return that string with number+units followed by the remaining string
    """
    numUnits = num_units(textstr)[0]
    parts = re.split(numUnits,textstr)
    for part in parts:
        if part:
            return "".join(num_units_to_words+' '+part.strip())
#
def convert_string_to_words(input_string):
    """
    Isolate in the input string into numerical value and unit.
    Convert the numerical value to words, followed by unit.
    Output the string in "numerical value in words"+ unit + quantity format
    """
    try:
        num_value,unit = num_units(input_string)[0].split(' ')
    except IndexError:
        return input_string
    except ValueError:
        print("\n***input_string = {}".format(input_string))
        return None
    # Convert the numerical value to words
    try:
        numerical_word = num2words(float(num_value))
    except ValueError:
        print("\n***num_value = {} input_string = {}".format(num_value,input_string))
        pass
            
    # Convert the unit to its word form

    unit_words = {
        "mg": "milligrams",
        "kg": "kilogram",
        "kgs":"kilograms",
        "kilo": "kilogram",
        "kilos":"kilograms",
        "L": "liters",
        "mL": "milliliters",
        "dozen": "dozen",
        "dozens": "dozen"
    }
    unit_word = unit_words.get(unit.strip(), "")
 
    # Construct the final output string
    try:
        num_units_words = f"{numerical_word} {unit_word}"
    except UnboundLocalError:
        print("input_string = {}".format(input_string))
    
    return isolate_num_units(input_string,num_units_to_words=num_units_words)


# In[ ]:


text1 = '1 dozens Gala Apples'
text2 = 'Gala Apples 2 dozens'
text3 = '1 kg flour'
text4 = 'tamrin 5kg live'
#
num_value,unit = num_units(text1)[0].split(' ')
numerical_word = num2words(float(num_value))
print(num_value,numerical_word)
#
print(num_units(text2))
print(num_units(text3))
print('num_units[0] ',num_units(text1)[0])


# In[ ]:


print(num_units(text4)[0].split(' '))
num_value,unit = num_units(text4)[0].split(' ')
print(num_units(num_value))
#convert_string_to_words(text4)


# In[ ]:


mystr = '1Kg'
number = re.findall(r'\d+', "1Kg")
print(number)
mystr[-(mystr[::-1].index(number[-1])):]


# In[ ]:


all_mp3_files = list_files_in_path(r'./data_mp3')
print(num_units('1 dozens Apples'))
print(num_units('1 kg Apples'))

print(convert_string_to_words('1 dozens Apples').lower())
print(convert_string_to_words('1 kg flour').lower())


# In[ ]:


pip install textdistance


# In[ ]:


from pathlib import Path, PureWindowsPath
import textdistance


# In[ ]:


#
model_id = 'whisper-1'
language = "En"
#
test_score = 0
#
for indx,eachf in enumerate(all_mp3_files[:]):
    each_mp3 = str(PureWindowsPath(eachf).as_posix())
    language='hi'
    audio_file= open(each_mp3, "rb")
    print("indx = {}, file = {}".format(indx, each_mp3))
    response = openai.Audio.translate("whisper-1",audio_file)
    """
    response = openai.Audio.transcribe(
        api_key=API_KEY,
        model=model_id,
        file=audio_file,
        language='en'
    )
    """
    #
    audio_transcript = (response['text'].lower()).strip()
    audio_transcript_text = audio_transcript.rstrip('.')
    #
    label = each_mp3.split('/')[-1].split('.')[0]
    label_text = (convert_string_to_words(label).lower()).strip()
    label_text = label_text.strip('.')
    #
    #print('label: ',label_text)
    #vectors = vectorizer.fit_transform([convert_string_to_words(label).lower(), audio_transcript['text'].lower()])
    #similarity = cosine_similarity(vectors)
    #
    try:
        audio_transcript_text = convert_string_to_words(audio_transcript_text)
    except:
        print('\n***IndexError in audio transcript:{}'.format(audio_transcript_text))
        pass
    # IndexError, ValueError:
    #
    try:
        similarity = textdistance.sorensen(label_text, audio_transcript_text)
    except AttributeError:
        print('\n*** AttributeError in label:{}'.format(label))
    #print('label = -{}- text = -{}-'.format(convert_string_to_words(label).lower(), audio_transcript['text'].lower()))
    if similarity > 0.8:
        test_score += 1
    else:
        print('\n\n***similarity = {} label = -{}- text = -{}-\n'.format(similarity,label_text, audio_transcript_text))
#
#


# In[ ]:


(309+24)/385.0


# In[ ]:


temp = (convert_string_to_words("20 litres cooking oil").lower()).strip()
print(temp)
textdistance.sorensen("1 kilos flour","1 kg flour")


# In[ ]:


print(float(test_score)/indx)


# In[ ]:


import textdistance
t1='one dozens apples'
t2= 'one dozen apples.'
print(textdistance.hamming(t1,t2))
print(textdistance.hamming.normalized_similarity(t1,t2))
print(textdistance.levenshtein.normalized_similarity(t1,t2))
print(textdistance.jaro_winkler(t1,t2))
print(textdistance.jaccard(t1 , t2))
print(textdistance.sorensen(t1,t2))
print(textdistance.ratcliff_obershelp(t1,t2))


# In[ ]:


import textdistance


# In[ ]:





# In[ ]:


print(textdistance.ratcliff_obershelp(text1, text2))
print(textdistance.levenshtein.normalized_similarity(text1, text2))


# # Jiju see here.

# ### Text conversion with num2words

# In[ ]:





# In[ ]:





# In[ ]:


text1 = "1 dozens Apples"
text2 = "One dozen apples"
# Convert here from above
print(convert_string_to_words(text1))


# In[ ]:





# In[ ]:


import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
# Convert the texts into TF-IDF vectors
vectorizer = TfidfVectorizer()
vectors = vectorizer.fit_transform([text1, text2])

# Calculate the cosine similarity between the vectors
similarity = cosine_similarity(vectors)
print(similarity)


# In[ ]:


# Test with "1 kg"
input_string = "1 kg"
output_string = convert_string_to_words(input_string)
print(output_string)  # Output: one kilogram

# Test with "200 mL"
input_string = "200 mL"
output_string = convert_string_to_words(input_string)
print(output_string)  # Output: two hundred milliliters


# In[ ]:





# In[ ]:





# PyTorch offers domain-specific libraries such as [TorchText](https://pytorch.org/text/stable/index.html),
# [TorchVision](https://pytorch.org/vision/stable/index.html), and [TorchAudio](https://pytorch.org/audio/stable/index.html),
# all of which include datasets. For this tutorial, we will be using a TorchVision dataset.
# 
# The ``torchvision.datasets`` module contains ``Dataset`` objects for many real-world vision data like
# CIFAR, COCO ([full list here](https://pytorch.org/vision/stable/datasets.html)). In this tutorial, we
# use the FashionMNIST dataset. Every TorchVision ``Dataset`` includes two arguments: ``transform`` and
# ``target_transform`` to modify the samples and labels respectively.

# In[ ]:


"""
# Download training data from open datasets
training_data = datasets.FashionMNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor(),
)

# Download test data from open datasets
test_data = datasets.FashionMNIST(
    root="data",
    train=False,
    download=True,
    transform=ToTensor(),
)
"""


# We pass the ``Dataset`` as an argument to ``DataLoader``. This wraps an iterable over our dataset, and supports
# automatic batching, sampling, shuffling and multiprocess data loading. Here we define a batch size of 64, i.e., each element
# in the dataloader iterable will return a batch of 64 features and labels.

# In[ ]:


"""
batch_size = 64

# Create data loaders
train_dataloader = DataLoader(training_data, batch_size=batch_size)
test_dataloader = DataLoader(test_data, batch_size=batch_size)

for X, y in test_dataloader:
    print("Shape of X [N, C, H, W]: ", X.shape)
    print("Shape of y: ", y.shape, y.dtype)
    break
"""


# Read more about [loading data in PyTorch](https://pytorch.org/tutorials/beginner/basics/data_tutorial.html).

# In[ ]:


## Creating models, including GPU

To define a neural network in PyTorch, we create a class that inherits
from [nn.Module](https://pytorch.org/docs/stable/generated/torch.nn.Module.html). We define the layers of the network
in the ``__init__`` function and specify how data will pass through the network in the ``forward`` function. To accelerate
operations in the neural network, we move it to the GPU if available.


# In[ ]:


"""
# Get cpu or gpu device for training
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using {} device".format(device))

# Define model
class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10)
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

model = NeuralNetwork().to(device)
print(model)
"""


# Read more about [building neural networks in PyTorch](https://pytorch.org/tutorials/beginner/basics/buildmodel_tutorial.html).

# ## Optimizing the model parameters
# 
# To train a model, we need a [loss function](https://pytorch.org/docs/stable/nn.html#loss-functions)
# and an [optimizer](https://pytorch.org/docs/stable/optim.html).

# In[ ]:


"""
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
"""


# In a single training loop, the model makes predictions on the training dataset (fed to it in batches), and
# backpropagates the prediction error to adjust the model's parameters.

# In[ ]:


"""
def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
"""


# We also check the model's performance against the test dataset to ensure it is learning.

# In[ ]:


def test(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")


# The training process is conducted over several iterations (*epochs*). During each epoch, the model learns
# parameters to make better predictions. We print the model's accuracy and loss at each epoch; we'd like to see the
# accuracy increase and the loss decrease with every epoch.

# In[ ]:


epochs = 5
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train(train_dataloader, model, loss_fn, optimizer)
    test(test_dataloader, model, loss_fn)
print("Done!")


# Read more about [Training your model](https://pytorch.org/tutorials/beginner/basics/optimization_tutorial.html).

# ## Saving models
# 
# A common way to save a model is to serialize the internal state dictionary (containing the model parameters).

# In[ ]:


"""
torch.save(model.state_dict(), "model.pth")
print("Saved PyTorch Model State to model.pth")
"""


# ## Loading models
# 
# The process for loading a model includes re-creating the model structure and loading
# the state dictionary into it.

# In[ ]:


"""
model = NeuralNetwork()
model.load_state_dict(torch.load("model.pth"))
""


# This model can now be used to make predictions.
# 
# 

# In[1]:


from datasets import Dataset
import pandas as pd
from datasets import Audio
import gc
import os
import pandas as pd

folder_path = '/notebooks/data_mp3/audio_p1'


# In[2]:


pip install -r requirementes.txt


# In[3]:


audio_files = []
labels = []


# In[4]:


for file_name in os.listdir(folder_path):
    audio_file_path = os.path.join(folder_path, file_name)
    audio_files.append(audio_file_path)
    label = file_name.split('.')[0]  # Assuming the label is the part before the extension
    labels.append(label)


# In[5]:


df = pd.DataFrame({'audio': audio_files, 'sentence': labels})


# In[6]:


df.head()


# In[7]:


from sklearn.model_selection import train_test_split

train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

train_df.to_csv('train.csv', index=False)
test_df.to_csv('test.csv', index=False)

train_dataset = Dataset.from_pandas(train_df)
test_dataset = Dataset.from_pandas(test_df)

train_dataset = train_dataset.cast_column("audio", Audio(sampling_rate=16000))
test_dataset= test_dataset.cast_column("audio", Audio(sampling_rate=16000))



# In[16]:


from transformers import WhisperFeatureExtractor
feature_extractor = WhisperFeatureExtractor.from_pretrained("openai/whisper-base")


# In[59]:


from transformers import WhisperTokenizer
tokenizer = WhisperTokenizer.from_pretrained("openai/whisper-base", language="English", task="transcribe")


# In[60]:


from transformers import WhisperProcessor
processor = WhisperProcessor.from_pretrained("openai/whisper-base", language="English", task="transcribe")


# In[61]:


def prepare_dataset(examples):
    # compute log-Mel input features from input audio array
    audio = examples["audio"]
    examples["input_features"] = feature_extractor(
        audio["array"], sampling_rate=16000).input_features[0]
    del examples["audio"]
    sentences = examples["sentence"]
    # encode target text to label ids
    examples["labels"] = tokenizer(sentences).input_ids
    del examples["sentence"]
    return examples


# In[62]:


train_dataset = train_dataset.map(prepare_dataset, num_proc=1)


# In[63]:


test_dataset = test_dataset.map(prepare_dataset, num_proc=1)


# In[65]:


import torch
from dataclasses import dataclass
from typing import Any, Dict, List, Union
@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any
    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # split inputs and labels since they have to be of different lengths and need different padding methods
        # first treat the audio inputs by simply returning torch tensors
        input_features = [{"input_features": feature["input_features"]} for feature in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")
        # get the tokenized label sequences
        label_features = [{"input_ids": feature["labels"]} for feature in features]
        # pad the labels to max length
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")
        # replace padding with -100 to ignore loss correctly
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)
        # if bos token is appended in previous tokenization step,
        # cut bos token here as it’s append later anyways
        if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all().cpu().item():
            labels = labels[:, 1:]
        batch["labels"] = labels
        return batch


# In[66]:


data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)


# In[67]:


import evaluate
metric = evaluate.load("wer")


# In[68]:


def compute_metrics(pred):
    pred_ids = pred.predictions
    label_ids = pred.label_ids
    # replace -100 with the pad_token_id
    label_ids[label_ids == -100] = tokenizer.pad_token_id
    # we do not want to group tokens when computing the metrics
    pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    label_str = tokenizer.batch_decode(label_ids, skip_special_tokens=True)
    wer = 100 * metric.compute(predictions=pred_str, references=label_str)
    return {"wer": wer}


# In[69]:


from transformers import WhisperForConditionalGeneration

model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-base")

model.config.forced_decoder_ids = None
model.config.suppress_tokens = []


# In[48]:





# In[70]:


from transformers import Seq2SeqTrainingArguments
training_args = Seq2SeqTrainingArguments(
    output_dir="./whisper-base-en",  # change to a repo name of your choice
    per_device_train_batch_size=16,
    gradient_accumulation_steps=1,  # increase by 2x for every 2x decrease in batch size
    learning_rate=1e-5,
    warmup_steps=500,
    max_steps=15000,
    gradient_checkpointing=True,
    fp16=True,
    evaluation_strategy="steps",
    per_device_eval_batch_size=1,
    predict_with_generate=True,
    generation_max_length=225,
    save_steps=500,
    eval_steps=500,
    # logging_steps=25,
    report_to=["tensorboard"],
    load_best_model_at_end=True,
    metric_for_best_model="wer",
    greater_is_better=False,
    push_to_hub=False,
)
trainer = Seq2SeqTrainer(
    args=training_args,
    model=model,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    tokenizer=processor.feature_extractor,
)


# In[2]:


pip install accelerate -U


# In[ ]:


"""
classes = [
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal","Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot",
]

model.eval()
x, y = test_data[0][0], test_data[0][1]
with torch.no_grad():
    pred = model(x)
    predicted, actual = classes[pred[0].argmax(0)], classes[y]
    print(f'Predicted: "{predicted}", Actual: "{actual}"')
"""


# Read more about [Saving & Loading your model](https://pytorch.org/tutorials/beginner/basics/saveloadrun_tutorial.html).

# ## Next steps
# 
# To proceed with PyTorch in Gradient, you can:
#     
#  - Look at other Gradient material, such as our [tutorials](https://docs.paperspace.com/gradient/tutorials/) and [blog](https://blog.paperspace.com)
#  - Try out further [PyTorch tutorials](https://pytorch.org/tutorials/beginner/basics/intro.html)
#  - Start writing your own projects, using our [documentation](https://docs.paperspace.com/gradient) when needed
#  
# If you get stuck or need help, [contact support](https://support.paperspace.com), and we will be happy to assist.
# 
# Good luck!

# ## Original PyTorch copyright notice
# 
# © Copyright 2021, PyTorch.
