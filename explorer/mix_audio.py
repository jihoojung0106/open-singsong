import librosa
import numpy as np
import soundfile as sf
import torchaudio
import torch
from torchaudio.functional import resample

import os
import sys
from pathlib import Path

import torch

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from open_musiclm.utils import int16_to_float32, float32_to_int16, zero_mean_unit_var_norm,exists

def vocal_preprocess(audio_path):
    
    data, sample_hz = torchaudio.load(audio_path)

    if data.shape[0] > 1:
        data = torch.mean(data, dim=0).unsqueeze(0)

    target_length = int(10 * sample_hz)

    data = data[:, :target_length]
    audio_for_encodec = resample(data, sample_hz, 24000)
    audio_for_encodec = int16_to_float32(float32_to_int16(audio_for_encodec))
    
def my_linear_mixing(audio1, audio2, output_file):
    # 오디오 파일 로드

    y1, sr1 = torchaudio.load(audio1)
    y2, sr2 = torchaudio.load(audio2)
    print("sr1",sr1,"sr2,",sr2)
    if y2.shape[0] > 1:
            y2 = torch.mean(y2, dim=0).unsqueeze(0)
    if y1.shape[0] > 1:
            y1 = torch.mean(y1, dim=0).unsqueeze(0)
     # 오디오 길이 일치화

    y2 = resample(y2, sr2, sr1)
    y2 = int16_to_float32(float32_to_int16(y2))
    
    min_length = min(y1.shape[1], y2.shape[1])

    print(min_length)
    y1 = y1[:,:min_length]
    print(y1.shape)
    y2 =y2[:, :min_length]
    print(y2.shape)
    # 선형으로 엮기
    mixed_audio = y1 + y2
    mixed_audio=mixed_audio.squeeze()
    # 결과 저장
    sf.write(output_file, mixed_audio, sr1)
    print(f"{output_file} 에 믹스 파일 저장함")

def linear_mixing(audio1, audio2, output_file):
    # 오디오 파일 로드
    y1, sr1 = librosa.load(audio1, sr=None)
    y2, sr2 = librosa.load(audio2, sr=None)
    # y3, sr3 = librosa.load(audio3, sr=None)

    # 오디오 길이 일치화
    min_length = min(len(y1), len(y2))
    y1 = y1[:min_length]
    y2 = y2[:min_length]
    # y3 = y3[:min_length]

    # 선형으로 엮기
    mixed_audio = y1 + y2
    # 결과 저장
    sf.write(output_file, mixed_audio, sr1)

if __name__ == '__main__':  
    inst="/content/drive/MyDrive/my_code/mymusiclm/my-open-musiclm-main/explorer/results_semcoarsetosem_super2/my_prediction_angel_0.wav"
    # voc="/content/drive/MyDrive/data/instrument/supersupermini/AClassic/vocals/vocals_1.wav"
    voc="/content/drive/MyDrive/data/instrument/supersupermini/Angela Thomas Wade - Milk Cow Blues/vocals/vocals_1.wav"
    gt_inst=voc.replace("vocals","instrument")
    folder_path="/content/drive/MyDrive/my_code/mymusiclm/my-open-musiclm-main/explorer/results_semcoarsetosem_super2"
    linear_mixing(gt_inst,voc,f"{folder_path}/gt_mix.wav")
    my_linear_mixing(inst,voc,f"{folder_path}/mix.wav")