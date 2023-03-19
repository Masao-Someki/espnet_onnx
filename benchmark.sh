#!/bin/bash

export PATH=/home/ubuntu/SCTK/bin:${PATH}

# full model
# Transformer CTC/Att

for onnx_type in full
do
	for model_type in Streaming
	do
		output_dir=tools/benchmark/result/${onnx_type}
		mkdir -p ${output_dir}/${model_type}
		python benchmark.py \
			tools.benchmark.config.${onnx_type}.${model_type} \
			${output_dir}/${model_type}

		# sclite \
		# 	-r ${output_dir}/${model_type}/ref.txt \
		# 	-h ${output_dir}/${model_type}/espnet_hypo.txt \
		# 	-i rm \
		# 	-o all stdout > ${output_dir}/${model_type}/espnet_wer.txt
		# python summarize.py \
		# 	${output_dir}/${model_type}/espnet_rtf.txt # > ${output_dir}/${model_type}/espnet_rtf.txt
		# python summarize.py \
		# 	${output_dir}/${model_type}/onnx_rtf.txt #> ${output_dir}/${model_type}/onnx_rtf.txt
	done
done

