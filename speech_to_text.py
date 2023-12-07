# -*- coding: utf-8 -*-
"""
Created on Wed Nov 22 13:18:10 2023

@author: DELINTE Nicolas
"""


import os
import torch
import librosa
import soundfile as sf
from tqdm import tqdm
from simple_diarizer.diarizer import Diarizer
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline


def speech_to_text(audio_file: str, num_speakers: int = 2,
                   language: str = 'french', path_to_ffmpeg: str = None):
    '''
    Takes in an audio file and returns a .txt with an approximative segmented
    transcription of the audio. The number of speakers must be specified.

    Parameters
    ----------
    audio_file : str
        Path to audio file.
    num_speakers : int, optional
        Number of people speaking in the recording. The default is 2.
    language : str, optional
        Name of the language. The default is 'french'.
    path_to_ffmpeg : str, optional
        Path to ffmpeg executable. Used for the file type conversion if not
        a .wav. The default is None.

    Returns
    -------
    None.

    '''

    # Adapt file type

    # https://www.gyan.dev/ffmpeg/builds/
    if 'wav' not in audio_file[-3:] and path_to_ffmpeg is not None:
        os.system(path_to_ffmpeg
                  + ' -i ' + audio_file+' ' +
                  audio_file[:-4]+'.wav')

    y, sr = librosa.load(audio_file[:-4]+'.wav')

    # Segment and classify

    diar = Diarizer(
        embed_model='ecapa',  # 'xvec' and 'ecapa' supported
        cluster_method='sc'  # 'ahc' and 'sc' supported
    )

    segments = diar.diarize(audio_file, num_speakers=num_speakers)

    segm_grouped = []
    v = segments[0]['label'].copy()
    segm_grouped.append(segments[0].copy())
    for i in range(len(segments)):
        v_s = segments[i]['label'].copy()
        if v == v_s:
            segm_grouped[-1]['end'] = segments[i]['end'].copy()
            segm_grouped[-1]['end_sample'] = segments[i]['end_sample'].copy()
        else:
            segm_grouped.append(segments[i].copy())
            v = v_s.copy()

    for i, s in enumerate(segm_grouped):

        if i == 0:
            start = 0
        else:
            start = int((s['start']+segm_grouped[i-1]['end'])/2*sr)
        if i == len(segm_grouped)-1:
            end = int(len(y))
        else:
            end = int((s['end']+segm_grouped[i+1]['start'])/2*sr)

        # Write out audio as 24bit PCM WAV
        sf.write(audio_file[:-4]+'_'+str(i)+'.wav', y[start:end], sr,
                 subtype='PCM_24')

    # Write segments to .txt

    device = "cpu"
    torch_dtype = torch.float32

    model_id = "openai/whisper-large-v3"

    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True,
        use_safetensors=True)
    model.to(device)

    processor = AutoProcessor.from_pretrained(model_id)

    pipe = pipeline("automatic-speech-recognition",
                    model=model,
                    tokenizer=processor.tokenizer,
                    feature_extractor=processor.feature_extractor,
                    max_new_tokens=128, chunk_length_s=30, batch_size=16,
                    return_timestamps=True,
                    torch_dtype=torch_dtype, device=device,
                    )

    with open(audio_file[:-4]+'.txt', 'w', encoding='latin1') as outfile:

        for i in tqdm(range(len(segm_grouped))):

            result = pipe(audio_file[:-4]+'_'+str(i)+'.wav',
                          generate_kwargs={"language": language})
            outfile.write('Personne '+str(segm_grouped[i]['label']+1))
            outfile.write('\n')
            outfile.write(result["text"])
            outfile.write('\n')

    return (result)
