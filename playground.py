import os

import torch

from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

import src.utils.fixseed

from src.datasets.get_dataset import get_datasets
from src.models.get_model import get_model as get_gen_model
from src.parser.training import parser
from src.train.trainer import train
from src.utils.get_model_and_data import get_model_and_data
from src.utils.tensors import collate


parameters = parser()
datasets = get_datasets(parameters)
dataset = datasets["train"]
# print(type(dataset[0]))
# print("dataset[0][0].shape", dataset[0][0].shape)
# print("dataset[0][1]", dataset[0][1])
train_iterator = DataLoader(
    dataset, batch_size=parameters["batch_size"], shuffle=True, num_workers=8, collate_fn=collate
)
for i, batch in tqdm(enumerate(train_iterator), desc="Computing batch"):
    # Put everything in device
    batch = {key: val.cuda() for key, val in batch.items()}
    for k, v in batch.items():
        print(k, v.shape)
    break

model = get_gen_model(parameters)

batch = model(batch)
mixed_loss, losses = model.compute_loss(batch)
