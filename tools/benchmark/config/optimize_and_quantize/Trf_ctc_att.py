import os
import glob
import tqdm

from espnet2.bin.asr_inference import Speech2Text as espnet_model
from espnet_onnx.utils.config import Config

tag_name = 'Shinji Watanabe/librispeech_asr_train_asr_transformer_e18_raw_bpe_sp_valid.acc.best'
device = 'cpu'
wav_dir = '/hdd/database/librispeech-100/test-clean/LibriSpeech/test-clean'
output_dir = './tools/benchmark/result/full/trf_ctc_att'

export = Config({
    'apply_quantize': True,
    'apply_optimize': True,
    'config': {
        'max_seq_len': 5000,
    },
})

espnet = Config({
    'model_config': {},
    'remove_modules': ['lm'],
    'require_inference': False
})

onnx = Config({
    'model_config': {
        'use_quantized': False
    },
    'remove_modules': ['lm']
})


def get_transcription(reader, chapter):
    ret = []
    trans_txt = glob.glob(os.path.join(wav_dir, reader, chapter, '*.trans.txt'))[0]
    with open(trans_txt, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    for l in lines:
        key = l.split(' ')[0]
        text = ' '.join(l.split(' ')[1:])
        ret.append({'key': key, 'text': text})
    return ret

def wav_loader():
    readers = glob.glob(os.path.join(wav_dir, '*'))[:1]
    for reader in readers:
        for chapter in glob.glob(os.path.join(wav_dir, reader, '*'))[:1]:
            # read transcription
            texts = get_transcription(reader, chapter)
            for d in tqdm.tqdm(texts[:5]):
                wav_file = os.path.join(wav_dir, reader, chapter, f'{d["key"]}.flac')
                utt_id = '_'.join(os.path.basename(wav_file).split('.')[0].split('-')[1:])
                utt_id = f'{os.path.basename(reader)}-{utt_id}'
                yield d['text'], wav_file, 16000, utt_id
            break

require_format = True
def format_hypo(text, utt_id):
    return f'({text}) ({utt_id})'
