import torch
from PerceiverAR import *
from dataset import *
from trainer import *

BLOCK_SIZE = 128
LATENT_SIZE = 32
CORPUS_PATH = "wiki.txt"
TEST_INDEX = 200

text = open(CORPUS_PATH, encoding="utf-8").read()
perceiver_dataset = PerceiverARDataset(text, BLOCK_SIZE, LATENT_SIZE)

model_config = GPTConfig(
    perceiver_dataset.vocab_size,
    perceiver_dataset.block_size,
    n_layer=4,
    n_head=8,
    n_embd=256,
    perceiver=True,
    latent_size=perceiver_dataset.latent_size,
    padding_idx=perceiver_dataset.stoi[perceiver_dataset.PAD_CHAR],
)

model = PerceiverAR(model_config)

def torch_to_string(x):
    return "".join([perceiver_dataset.itos[i] for i in x.numpy()])

x, y = perceiver_dataset[TEST_INDEX]
model.eval()
y_pred = model(x.unsqueeze(0))
max_indices = y_pred.argmax(dim=2)

x_string = torch_to_string(x)[:-LATENT_SIZE+1]
y_string = torch_to_string(y)
y_pred_string = torch_to_string(max_indices.squeeze(0))

print(f"Start - {x_string}")
print(f"Target - {y_string}")
print(f"Predicted - {y_pred_string}")