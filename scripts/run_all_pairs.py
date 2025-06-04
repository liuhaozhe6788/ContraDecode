import itertools
import subprocess
import json
import csv
import argparse
from tqdm import tqdm
import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from data_preparation.preprocessing import lang_pairs


def main(args):
    # Output directory
    model_name = args.model_name
    for lang_pair in tqdm(lang_pairs[:10]):
        src_lang, tgt_lang = lang_pair
        lang_pair = src_lang + "-" + tgt_lang
        # create dataset
        try:
            run_cmd = f"python -m data_preparation.create_customized_dataset \
                --language_pair {lang_pair}"
            subprocess.run(run_cmd, shell=True, check=True)
        except Exception as e:
            print(f"❌ Create dataset for {lang_pair} failed: {e}")
        # Run generation
        try:
            run_cmd = f"python -m scripts.run \
                --model_path {model_name} \
                --language_pairs {lang_pair} \
                --use_customized"
            subprocess.run(run_cmd, shell=True, check=True)
        except Exception as e:
            print(f"❌ Direct generation for {lang_pair} failed: {e}")
        try:
            run_cmd = f"python -m scripts.run \
                --model_path {model_name} \
                --language_pairs {lang_pair} \
                --source_contrastive 2 \
                --source_weight -0.7 \
                --language_contrastive en src \
                --language_weight -0.1 \
                --use_customized"
            subprocess.run(run_cmd, shell=True, check=True)
        except Exception as e:
            print(f"❌ Source language contrastive generation for {lang_pair} failed: {e}")
        try:
            run_cmd = f"python -m scripts.run \
                --model_path {model_name}_tr_st \
                --language_pairs {lang_pair} \
                --model_contrastive \
                --student_coef 0.05 \
                --student_temperature 1 \
                --use_dynamic_coef \
                --use_customized"
            subprocess.run(run_cmd, shell=True, check=True)
        except Exception as e:
            print(f"❌ Model contrastive decoder-only generation for {lang_pair} failed: {e}")
        try:
            run_cmd = f"python -m scripts.run \
                --model_path {model_name}_tr_st \
                --language_pairs {lang_pair} \
                --model_contrastive \
                --student_coef 0.05 \
                --student_temperature 1 \
                --student_model_type attention_scaling \
                --attention_scale 0.25 \
                --use_dynamic_coef \
                --use_customized"
            subprocess.run(run_cmd, shell=True, check=True)
        except Exception as e:
            print(f"❌ Model contrastive attention scaling generation for {lang_pair} failed: {e}")
        try:
            run_cmd = f"python -m scripts.run \
                --model_path {model_name}_hybrid \
                --language_pairs {lang_pair} \
                --source_contrastive 2 \
                --source_weight -0.1 \
                --language_contrastive en src \
                --language_weight -0.1 \
                --model_contrastive \
                --student_coef 0.05 \
                --student_temperature 1 \
                --use_dynamic_coef \
                --student_alpha 0.01 \
                --use_customized"
            subprocess.run(run_cmd, shell=True, check=True)
        except Exception as e:
            print(f"❌ Hybrid contrastive decoder-only generation for {lang_pair} failed: {e}")
        try:
            run_cmd = f"python -m scripts.run \
                --model_path {model_name}_hybrid \
                --language_pairs {lang_pair} \
                --source_contrastive 2 \
                --source_weight -0.1 \
                --language_contrastive en src \
                --language_weight -0.1 \
                --model_contrastive \
                --student_coef 0.05 \
                --student_temperature 1 \
                --use_dynamic_coef \
                --student_model_type attention_scaling \
                --attention_scale 0.25 \
                --student_alpha 0.01 \
                --use_customized"
            subprocess.run(run_cmd, shell=True, check=True)
        except Exception as e:
            print(f"❌ Hybrid contrastive attention scaling generation for {lang_pair} failed: {e}")
    
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, required=True, choices=["small100", "m2m100_418M"],
                        help="The translation model name")
    args = parser.parse_args()
    main(args)