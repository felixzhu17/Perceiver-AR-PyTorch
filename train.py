import torch
from PerceiverAR import *
from dataset import *
from trainer import *

BLOCK_SIZE = 128
LATENT_SIZE = 32
CORPUS_PATH = "wiki.txt"

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

trainer_config = TrainerConfig(
    max_epochs=1,
    lr_decay=True,
    ckpt_path="perceiver_ar.pt",
)
model.load_state_dict(torch.load(trainer_config.ckpt_path))
trainer = Trainer(model, perceiver_dataset, trainer_config)
trainer.train()