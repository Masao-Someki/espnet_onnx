
import argparse
import shutil
import logging
import os
import time

import librosa

from espnet_onnx import Speech2Text
from espnet_onnx.export import ASRModelExport
from importlib import import_module

logger = logging.getLogger(__name__)


def maybe_remove_modules(model, module_list):
    removed = False
    for m in module_list:
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


def measure_time(model, wav, sr):
    start_time = time.time()
    result_text = model(y)[0][0]
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
    args = parser.parse_args()

    # load config file
    config = import_module(args.config_module)

    # export onnx model
    m = ASRModelExport()
    m.set_export_config(**config.export.config)
    m.export_from_pretrained(
        config.tag_name,
        quantize=config.export.apply_quantize,
        optimize=config.export.apply_optimize
    )

    # load espnet model
    if config.espnet.require_inference:
        espnet_model = config.espnet_model.from_pretrained(
            config.tag_name,
            device=config.device,
            **config.espnet.model_config
        )

    # load onnx model
    PROVIDER = 'CUDAExecutionProvider' if config.device == 'cuda' else 'CPUExecutionProvider'
    onnx_model = Speech2Text(
        config.tag_name,
        providers=[PROVIDER],
        **config.onnx.model_config
    )

    # remove lm/decoder if required
    maybe_remove_modules(espnet_model, config.espnet.remove_modules)
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

        if config.espnet.require_inference:
            rtf_e, res_e = measure_time(espnet_model, y, sr)
        else:
            rtf_e = None
            res_e = None
            
        rtf_o, res_o = measure_time(onnx_model, y, sr)

        if config.require_format:
            res_e = config.format_hypo(res_e, wav_id)
            res_o = config.format_hypo(res_o, wav_id)

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
    write_result(config.output_dir, 'espnet_rtf', rtfs_e)
    write_result(config.output_dir, 'espnet_hypo', results_e)
    write_result(config.output_dir, 'onnx_rtf', rtfs_o)
    write_result(config.output_dir, 'onnx_hypo', results_o)
    write_result(config.output_dir, 'ref', results_o)
