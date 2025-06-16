import os
import itertools
import subprocess
import json
import csv
import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

# Output directory
results_dir = "out/flores/af-zu"
os.makedirs(results_dir, exist_ok=True)

# Tuning space (only contrastive-related weights)
student_coefs = [0.3, 0.5, 0.7]
student_min_probs = [0.0]
student_temperatures = [0.5, 1.0]
source_weights = [-0.3, -0.7]
language_weights = [-0.1, -0.3]

# Fixed settings
language_pair = "af-zu"
ref_file = os.path.join(results_dir, "ref.txt")
model_path = "small100_hybrid"
out_file_name = "hybrid_contrastive_decoder_only.txt"

# CSV output
csv_file = "tuning_results.csv"
with open(csv_file, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow([
        "student_coef", "student_min_prob", "student_temperature",
        "source_weight", "language_weight",
        "chrF2", "BLEU"
    ])

    for student_coef, min_prob, temp, src_w, lang_w in itertools.product(
        student_coefs, student_min_probs, student_temperatures, source_weights, language_weights
    ):
        print(f"\n🔧 Running: student_coef={student_coef}, temp={temp}, sw={src_w}, lw={lang_w}")
        out_file = os.path.join(results_dir, out_file_name)

        # Run generation
        run_cmd = f"""python -m scripts.run \
            --model_path {model_path} \
            --language_pairs {language_pair} \
            --source_contrastive 2 \
            --source_weight {src_w} \
            --language_contrastive en ps \
            --language_weight {lang_w} \
            --model_contrastive \
            --student_coef {student_coef} \
            --student_min_prob {min_prob} \
            --use_dynamic_coef \
            --student_temperature {temp} \
            --use_customized"""
        subprocess.run(run_cmd, shell=True, check=True)

        # Evaluate
        try:
            chrf_raw = subprocess.check_output(
                f"sacrebleu {ref_file} < {out_file} --metrics chrf", shell=True, text=True)
            bleu_raw = subprocess.check_output(
                f"sacrebleu {ref_file} < {out_file} --tokenize flores101", shell=True, text=True)
        
            # Extract scores from plain output
            chrf_score = eval(chrf_raw)['score']
            bleu_score = eval(bleu_raw)['score']
        except Exception as e:
            print(f"❌ Evaluation failed: {e}")
            chrf_score = bleu_score = -1

        writer.writerow([student_coef, min_prob, temp, src_w, lang_w, chrf_score, bleu_score])
        f.flush()
        print(f"✅ chrF2: {chrf_score:.2f}, BLEU: {bleu_score:.2f}")
