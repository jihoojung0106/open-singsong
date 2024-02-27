'''
1) load audio samples
2) compute clap tokens and semantic tokens
3) run them through coarse stage to predict coarse tokens
4) reconstruct audio from coarse tokens
Reconstructed audio should be semantically similar to the original audio if hubert-kmeans and coarse stage are working correctly

example usage:

python scripts/infer_coarse.py \
    ./data/fma_large/000/000005.mp3 \
    ./data/fma_large/000/000010.mp3 \
    --model_config ./configs/model/musiclm_small.json \
    --coarse_path ./results/coarse_continue_1/coarse.transformer.10000.pt

'''

import argparse
import os
import sys
from pathlib import Path

import torch
import torchaudio
from einops import rearrange
from torchaudio.functional import resample

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from open_musiclm.config import (create_clap_quantized_from_config,
                                 create_coarse_transformer_from_config,
                                 create_semcoarsetosem_transformer_from_config,
                                 create_encodec_from_config,
                                 create_hubert_kmeans_from_config,
                                 my_load_model_config,load_model_config)
from open_musiclm.open_musiclm import (SemcoarsetosemStage,CoarseStage,
                                       get_or_compute_clap_token_ids,
                                       get_or_compute_acoustic_token_ids,
                                       get_or_compute_semantic_token_ids)
from open_musiclm.utils import int16_to_float32, float32_to_int16, zero_mean_unit_var_norm
from scripts.train_utils import disable_print


def prepare_inference(audio_files,model_config,model_path):
    
    model_config = my_load_model_config(model_config)

    # coarse_path = "/content/drive/MyDrive/my_code/mymusiclm/my-open-musiclm-main/explorer/wandb/run-20240223_102209-e01qb72w/files/coarse_generation_test.transformer.900.pt"
    rvq_path="/content/drive/MyDrive/data/weight/clap.rvq.950_no_fusion.pt"
    kmeans_path="/content/drive/MyDrive/data/weight/kmeans_10s_no_fusion.joblib"

    seed = 42

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    clap = create_clap_quantized_from_config(model_config, rvq_path, device)
    wav2vec = create_hubert_kmeans_from_config(model_config, kmeans_path, device)
    encodec_wrapper = create_encodec_from_config(model_config, device)
    semcoarsetosem_transformer = create_semcoarsetosem_transformer_from_config(model_config, model_path, device)

    semcoarsetosem_stage = SemcoarsetosemStage(
        semcoarsetosem_transformer=semcoarsetosem_transformer,
        neural_codec=encodec_wrapper,
        wav2vec=wav2vec,
    )

    torch.manual_seed(42)

    
    audios_for_wav2vec = [] #semantic
    audios_for_encodec = [] #coarse
    for audio_path in audio_files:
        data, sample_hz = torchaudio.load(audio_path)

        if data.shape[0] > 1:
            data = torch.mean(data, dim=0).unsqueeze(0)

        target_length = int(10 * sample_hz)
        normalized_data = zero_mean_unit_var_norm(data)

        data = data[:, :target_length]
        normalized_data = normalized_data[: , :target_length]
        audio_for_encodec = resample(data, sample_hz, encodec_wrapper.sample_rate)
        audio_for_wav2vec = resample(normalized_data, sample_hz, wav2vec.target_sample_hz)

        audio_for_encodec = int16_to_float32(float32_to_int16(audio_for_encodec))
        audio_for_wav2vec = int16_to_float32(float32_to_int16(audio_for_wav2vec))

        audios_for_encodec.append(audio_for_encodec)
        audios_for_wav2vec.append(audio_for_wav2vec)

    audios_for_encodec = torch.cat(audios_for_encodec, dim=0).to(device)
    audios_for_wav2vec = torch.cat(audios_for_wav2vec, dim=0).to(device)

    semantic_token_ids = get_or_compute_semantic_token_ids(None, audios_for_wav2vec, wav2vec)
    coarse_token_ids, fine_token_ids = get_or_compute_acoustic_token_ids(None, None, audios_for_encodec, encodec_wrapper, model_config.global_cfg.num_coarse_quantizers)
    vocals_semantic_token_ids=semantic_token_ids[0].unsqueeze(0)
    vocals_coarse_token_ids=coarse_token_ids[0].unsqueeze(0)
    other_semantic_token_ids=semantic_token_ids[1].unsqueeze(0)
    other_coarse_token_ids=coarse_token_ids[1].unsqueeze(0)
    
    return semcoarsetosem_stage,vocals_semantic_token_ids,vocals_coarse_token_ids,other_semantic_token_ids

  
def make_audios(semantic_token_ids,results_folder,name=None):
    model_config="/content/drive/MyDrive/my_code/mymusiclm/open-musiclm-main/configs/model/musiclm_large_small_context.json"
    model_config = load_model_config(model_config)
    duration=5
    Path(results_folder).mkdir(parents=True, exist_ok=True)
    
    coarse_path="/content/drive/MyDrive/data/weight/coarse.transformer.18000.pt"
    rvq_path="/content/drive/MyDrive/data/weight/clap.rvq.950_no_fusion.pt"
    kmeans_path="/content/drive/MyDrive/data/weight/kmeans_10s_no_fusion.joblib"
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    clap = create_clap_quantized_from_config(model_config,rvq_path, device)
    wav2vec = create_hubert_kmeans_from_config(model_config, kmeans_path, device)
    encodec_wrapper = create_encodec_from_config(model_config, device)
    coarse_transformer = create_coarse_transformer_from_config(model_config, coarse_path, device)
    torch.manual_seed(42)
    coarse_stage = CoarseStage(
            coarse_transformer=coarse_transformer,
            neural_codec=encodec_wrapper,
            wav2vec=wav2vec,
            clap=clap
        )
    text=["many different kind of instrument, rich"]
    clap_token_ids = get_or_compute_clap_token_ids(None, clap, None, text)
    generated_wave = coarse_stage.generate(
        clap_token_ids=clap_token_ids,
        semantic_token_ids=semantic_token_ids,
        #gt_semantic_token_ids[0].unsqueeze(0),
        coarse_token_ids=None,
        max_time_steps=duration*75,
        # max_time_steps=int(model_config.global_cfg.coarse_audio_length_seconds * 75),
        reconstruct_wave=True,
        include_eos_in_output=False,
        append_eos_to_conditioning_tokens=True,
        temperature=0.95,
    )
    
    generated_wave = rearrange(generated_wave, 'b n -> b 1 n').detach().cpu()

    for i, wave in enumerate(generated_wave):
        torchaudio.save(f'{results_folder}/{name}_{i}.wav', wave, encodec_wrapper.sample_rate)
        print("=============================================================")
        print(f"{results_folder}/{name}_{i}.wav 에 semacoarsetosem_generation\n\n\n")


def return_token_ids(semcoarsetosem_stage,vocals_semantic_token_ids,vocals_coarse_token_ids):
    token_ids = semcoarsetosem_stage.generate(
        vocals_semantic_token_ids=vocals_semantic_token_ids,
        vocals_coarse_token_ids=vocals_coarse_token_ids,
        max_time_steps=200,
        temperature=0.95,
    )
    return token_ids


if __name__ == '__main__':    
    audio_files = ["/content/drive/MyDrive/data/instrument/supersupermini/AClassic/vocals/vocals_1.wav",
                   "/content/drive/MyDrive/data/instrument/supersupermini/AClassic/instrument/instrument_1.wav"]
    model_path="real_semcoarsetosem.transformer.5160.pt"
    model_config="my_musiclm_for_semcoarsetosem.json"
    
    semcoarsetosem_stage,vocals_semantic_token_ids,vocals_coarse_token_ids,other_semantic_token_ids = prepare_inference(audio_files,model_config,model_path)
    generated_semantic_token_ids=return_token_ids(semcoarsetosem_stage,vocals_semantic_token_ids,vocals_coarse_token_ids)
    results_folder = "/content/drive/MyDrive/my_code/mymusiclm/my-open-musiclm-main/explorer/semcoarsetosem_generation"
    make_audios(generated_semantic_token_ids.squeeze(2),results_folder,"aclasiic_my_prediction")
    