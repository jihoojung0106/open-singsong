import argparse
import os
import sys
from pathlib import Path

import torch

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from open_musiclm.config import (create_clap_quantized_from_config,
                                 create_semcoarsetosem_transformer_from_config,
                                 create_encodec_from_config,
                                 create_hubert_kmeans_from_config,
                                 my_create_single_stage_trainer_from_config,
                                 my_load_model_config, my_load_training_config)
from scripts.train_utils import load_checkpoint_from_args, validate_train_args

model_config="my_musiclm_for_semcoarsetosem.json"
model_config = my_load_model_config(model_config)
training_config="my_train_musiclm_fma.json"
training_config = my_load_training_config(training_config)

device = 'cuda' if torch.cuda.is_available() else 'cpu'

use_preprocessed_data = False

rvq_path="/content/drive/MyDrive/data/weight/clap.rvq.950_no_fusion.pt"
kmeans_path="/content/drive/MyDrive/data/weight/kmeans_10s_no_fusion.joblib"
clap = None
wav2vec = create_hubert_kmeans_from_config(model_config, kmeans_path, device)
encodec_wrapper = create_encodec_from_config(model_config, device)
results_folder="/content/drive/MyDrive/my_code/mymusiclm/my-open-musiclm-main/explorer/real_semcoarsetosem"
print('loading coarse stage...')
fine_tune_from=None
# path="/content/drive/MyDrive/my_code/mymusiclm/my-open-musiclm-main/explorer/wandb/run-20240223_082923-qjtu0sig/files/coarse_generation_test.transformer.500.pt"
semcoarsetosem_transformer = create_semcoarsetosem_transformer_from_config(model_config, fine_tune_from, device)

if not os.path.exists(results_folder):
    os.makedirs(results_folder)
    print(f"폴더 '{results_folder}'가 생성되었습니다.")
else:
    print(f"폴더 '{results_folder}'가 이미 존재합니다.")

trainer = my_create_single_stage_trainer_from_config(
    model_config=model_config,
    training_config=training_config,
    stage='semcoarsetosem',
    results_folder=results_folder,
    transformer=semcoarsetosem_transformer,
    clap=clap,
    wav2vec=wav2vec,
    encodec_wrapper=encodec_wrapper,
    device=device,
    use_wandb_tracking=True,
    wandb_name="real_semcoarsetosem",
    config_paths=[model_config, training_config])
    
train_path="real_semcoarsetosem.transformer.5150.pt"
op_path=train_path.replace("transformer","optimizer")
sch_path=train_path.replace("transformer","scheduler")
step=int(train_path.split(".")[-2])
trainer.load(train_path,op_path,sch_path,step)
trainer.train()
