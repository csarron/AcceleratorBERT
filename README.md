# Usage

````bash
python export_model.py -c bert_config.json -o data/bert_base -l 384

python3 mo_tf.py --input_meta_graph data/bert_base/model.meta \
--output prob \
--disable_nhwc_to_nchw \
--input input_ids{i32},segment_ids{i32}

python run_ncs.py -m model.xml

python run_tflite.py -m data/bert_base/model.tflite

# export bert_base for different input size
cd /opt/intel/openvino/deployment_tools/model_optimizer/
src_dir="this source code path"
for size in `seq 510 -20 10` ; do
  echo "running ${size}..."
  python3 "${src_dir}/run_ckpt_model.py" -c "${src_dir}/bert_config.json" -l ${size} 2>&1 | tee "${src_dir}/data/bert_base/tf_seq${size}.log"

  python3 mo_tf.py --input_meta_graph "data/bert_base/model_seq${size}.meta" \
  --output prob \
  --disable_nhwc_to_nchw \
  --input input_ids{i32},segment_ids{i32},input_mask{f32} \
  --progress --output_dir data/bert_base_ncs

  python3 "${src_dir}/run_ncs.py" -m data/bert_base_ncs/model_seq${size}.xml -s ${size} 2>&1 | tee "data/bert_base_ncs_seq${size}.log"
  rm -rf "data/bert_base/model_seq*"
  echo `pwd`;
done

# only run the converted models
# -d MYRIAD.26.2-ma2450 for ncs1
# -d MYRIAD.26.3-ma2480 for ncs2
for size in `seq 510 -20 10` ; do
  echo "running ${size}..."
  python3 "${src_dir}/run_ncs.py" -m data/bert_base_ncs/model_seq${size}.xml -s ${size} 2>&1 | tee "data/bert_base_ncs_seq${size}.log"
  echo `pwd`;
done

# for bert 24 models
for a in 2 4 8 12; do
  h=$((a*64))
  for l in $(seq 2 2 12); do
    for size in 100 ; do
      m=l${l}_h${h}_s${size}
      echo "running m=${m}..."

      python3 "${src_dir}/run_ckpt_model.py" -c data/bert_24/uncased_L-${l}_H-${h}_A-${a}/bert_config.json -cf data/bert_24/uncased_L-${l}_H-${h}_A-${a}/bert_model.ckpt -od data/bert_24 -on model_${m} -l ${size} 2>&1 | tee data/bert_24/tf_${m}.log

      python3 mo_tf.py --input_meta_graph data/bert_24/model_${m}.meta \
      --output prob \
      --disable_nhwc_to_nchw \
      --input input_ids{i32},segment_ids{i32},input_mask{f32} \
      --progress --output_dir data/bert_24_ncs

      python3 "${src_dir}/run_ncs.py" -m data/bert_24_ncs/model_${m}.xml -s ${size} 2>&1 | tee data/bert_24_ncs_${m}.log

    done
  done
done


for a in 2 4 8 12; do
  h=$((a*64))
  for l in $(seq 2 2 12); do
    for size in 100 ; do
      m=l${l}_h${h}_s${size}
      for d in ncs1 ncs2; do
        echo "running ${m} on ${d}..."
        python3 "${src_dir}/run_ncs.py" -m data/bert_24_ncs/model_${m}.xml -s ${size} -d ${d} 2>&1 | tee data/bert_${d}_${m}.log
      done
    done
  done
done
````
