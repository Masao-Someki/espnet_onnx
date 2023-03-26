
import argparse
import shutil
import logging
import os
import time

import torch
import librosa

from espnet_onnx.asr.beam_search.beam_search_transducer import BeamSearchTransducer
from espnet_onnx import StreamingSpeech2Text
from espnet_onnx.export import ASRModelExport
from importlib import import_module

logger = logging.getLogger(__name__)

def maybe_quantize_module(model):
    model.st_model.encoder = torch.quantization.quantize_dynamic(
        model.st_model.encoder,  # the original model
        {torch.nn.Linear},  # a set of layers to dynamically quantize
        dtype=torch.qint8)  # the target dtype for quantized weights
    dec = torch.quantization.quantize_dynamic(
        model.beam_search.scorers['decoder'],  # the original model
        {torch.nn.Linear},  # a set of layers to dynamically quantize
        dtype=torch.qint8)  # the target dtype for quantized weights
    model.beam_search.scorers['decoder'] = dec
    model.beam_search.full_scorers['decoder'] = dec

def maybe_remove_modules(model, module_list):
    removed = False
    for m in module_list:
        if model.beam_search is None:
            model.beam_search_transducer.lm = None
            model.beam_search_transducer.use_lm = False
            continue
        elif isinstance(model.beam_search, BeamSearchTransducer):
            model.beam_search.lm = None
            model.beam_search.lm = None
            continue

        if m in model.beam_search.scorers.keys():
            removed = True
            del model.beam_search.scorers[m]
        if m in model.beam_search.full_scorers.keys():
            removed = True
            del model.beam_search.full_scorers[m]
        if m in model.beam_search.part_scorers.keys():
            removed = True
            del model.beam_search.part_scorers[m]
        if not removed:
            logger.info(f'There is no {m} in the model.')


def run_model(model, y, use_simulate=False):
    if use_simulate:
        result_text = model.simulate(y)[0][0]
    else:
        result_text = model(y)[0][0]
    return result_text


def measure_time(model, wav, sr, use_simulate=False):
    start_time = time.time()
    result_text = run_model(model, wav, use_simulate)
    elapsed_time = time.time() - start_time
    wav_length = len(wav) / sr
    rtf = elapsed_time / wav_length
    return rtf, result_text


def write_result(output_dir, file_name, l):
    with open(os.path.join(output_dir, f'{file_name}.txt'), 'w', encoding='utf-8') as f:
        f.write('\n'.join(l))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('config_module', type=str, help='Directory to the wav file.')
    parser.add_argument('output_dir', type=str)
    args = parser.parse_args()

    # load config file
    config = import_module(args.config_module)

    # load espnet model
    if config.espnet.require_inference:
        # try:
        #     espnet_model = config.espnet_model.from_pretrained(
        #         config.tag_name,
        #         device=config.device,
        #         beam_size=10,
        #         **config.espnet.model_config.dic
        #     )
        # except:
        #     espnet_model = config.espnet_model.from_pretrained(
        #         config.tag_name,
        #         device=config.device,
        #         beam_size=10,
        #         **config.espnet.model_config.dic
        #     )
        from espnet2.bin.asr_inference_streaming import Speech2TextStreaming
        espnet_model = Speech2TextStreaming(
                asr_train_config='/home/masao/doc/espnet_onnx/tools/venv/lib64/python3.8/site-packages/espnet_model_zoo/models--D-Keqi--espnet_asr_train_asr_streaming_transformer_raw_en_bpe500_sp_valid.acc.ave/snapshots/948310c11d224280984270ae4a3e9792b2fbffce/exp/asr_train_asr_streaming_transformer_raw_en_bpe500_sp/config.yaml',
                asr_model_file='/home/masao/doc/espnet_onnx/tools/venv/lib64/python3.8/site-packages/espnet_model_zoo/models--D-Keqi--espnet_asr_train_asr_streaming_transformer_raw_en_bpe500_sp_valid.acc.ave/snapshots/948310c11d224280984270ae4a3e9792b2fbffce/exp/asr_train_asr_streaming_transformer_raw_en_bpe500_sp/valid.acc.ave_10best.pth',
                lm_train_config='/home/masao/doc/espnet_onnx/tools/venv/lib64/python3.8/site-packages/espnet_model_zoo/models--D-Keqi--espnet_asr_train_asr_streaming_transformer_raw_en_bpe500_sp_valid.acc.ave/snapshots/948310c11d224280984270ae4a3e9792b2fbffce/exp/lm_train_lm_en_bpe500/config.yaml',
                lm_file='/home/masao/doc/espnet_onnx/tools/venv/lib64/python3.8/site-packages/espnet_model_zoo/models--D-Keqi--espnet_asr_train_asr_streaming_transformer_raw_en_bpe500_sp_valid.acc.ave/snapshots/948310c11d224280984270ae4a3e9792b2fbffce/exp/lm_train_lm_en_bpe500/19epoch.pth',
                token_type='bpe',
                bpemodel='/home/masao/.cache/espnet_onnx/D-Keqi/espnet_asr_train_asr_streaming_transformer_raw_en_bpe500_sp_valid.acc.ave/bpe.model',
                beam_size=20,
                maxlenratio=0.0,
                minlenratio=0.0,
                ctc_weight=0.3,
                lm_weight=1.0,
                nbest=1,
                disable_repetition_detection=False,
                #decoder_text_length_limit=decoder_text_length_limit,
                #encoded_feat_length_limit=encoded_feat_length_limit,
                device='cuda'
            )
        maybe_remove_modules(espnet_model, config.espnet.remove_modules)
        # maybe_quantize_module(espnet_model)
    
    # export onnx model
    m = ASRModelExport()
    m.set_export_config(**config.export.config)
    m.export_from_pretrained(
        config.tag_name,
        quantize=config.export.apply_quantize,
        optimize=config.export.apply_optimize,
        pretrained_config=config.espnet.model_config.dic
    )

    # load espnet model
    # if config.espnet.require_inference:
    #     espnet_model = config.espnet_model.from_pretrained(
    #         config.tag_name,
    #         device=config.device,
    #         **config.espnet.model_config
    #     )
    #     maybe_remove_modules(espnet_model, config.espnet.remove_modules)


    # load onnx model
    PROVIDER = 'CUDAExecutionProvider' if config.device == 'cuda' else 'CPUExecutionProvider'
    onnx_model = StreamingSpeech2Text(
        config.tag_name.replace('.zip', ''),
        providers=[PROVIDER],
        **config.onnx.model_config
    )
    onnx_model.beam_search.beam_size = 10

    # remove lm/decoder if required
    maybe_remove_modules(onnx_model, config.onnx.remove_modules)

    # load wav files and write output
    is_first_iter = True
    # shutil.rmtree(config.output_dir)
    os.makedirs(config.output_dir, exist_ok=True)

    rtfs_e = []
    rtfs_o = []
    results_e = []
    results_o = []
    refs = []

    for ref_text, wav_file, sr, wav_id in config.wav_loader():
        y, sr = librosa.load(wav_file, sr=sr)
        # y = wav_file

        if config.espnet.require_inference:
            rtf_e, res_e = measure_time(espnet_model, y, sr)

        else:
            rtf_e = None
            res_e = None
            
        rtf_o, res_o = measure_time(onnx_model, y, sr, use_simulate=True)

        if config.require_format:
            res_e = config.format_hypo(res_e, wav_id)
            res_o = config.format_hypo(res_o, wav_id)
            ref_text = config.format_hypo(ref_text, wav_id)

        if is_first_iter:
            is_first_iter = False
            continue
        
        if config.espnet.require_inference:
            rtfs_e.append(str(rtf_e))
            results_e.append(res_e)

        rtfs_o.append(str(rtf_o))
        results_o.append(res_o)
        refs.append(ref_text)
        
    
    # write result into file
    write_result(args.output_dir, 'espnet_rtf', rtfs_e)
    write_result(args.output_dir, 'espnet_hypo', results_e)
    write_result(args.output_dir, 'onnx_rtf', rtfs_o)
    write_result(args.output_dir, 'onnx_hypo', results_o)
    write_result(args.output_dir, 'ref', refs)