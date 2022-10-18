#!/bin/bash


# full model
# Transformer CTC/Att

for onnx_type in full quantize optimize optimize_and_quantize
do
	for model_type in Trf_ctc_att Cfm_ctc_att Cfm_hubert
	do
		rm -r /home/ubuntu/.cache/espnet_onnx
		output_dir=tools/benchmark/result/${onnx_type}
		#python benchmark.py tools.benchmark.config.${onnx_type}.${model_type} ${output_dir}
		mkdir -p ${output_dir}/result/${model_type}
		sclite \
			-r ${output_dir}/ref.txt \
			-h ${output_dir}/onnx_hypo.txt \
			-i rm \
			-o all stdout > ${output_dir}/result/${model_type}/onnx_wer.txt
		python summarize.py \
			${output_dir}/espnet_rtf.txt > ${output_dir}/result/${model_type}/espnet_rtf.txt
		python summarize.py \
			${output_dir}/onnx_rtf.txt > ${output_dir}/result/${model_type}/onnx_rtf.txt
	done
done

