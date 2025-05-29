import os
import itertools
import subprocess
import json
import csv

# Output directory
results_dir = "out/flores/af-zu"
os.makedirs(results_dir, exist_ok=True)

# Tuning space (only contrastive-related weights)
st_coefs = [0.3, 0.5, 0.7]
student_min_probs = [0.0]
student_temperatures = [0.5, 1.0]
source_weights = [-0.3, -0.7]
language_weights = [-0.1, -0.3]

# Fixed settings
language_pair = "af-zu"
ref_file = os.path.join(results_dir, "ref.txt")
model_path = "small100_hybrid"
out_file_name = "hybrid-contrastive.txt"

# CSV output
csv_file = "tuning_results.csv"
with open(csv_file, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow([
        "st_coef", "student_min_prob", "student_temperature",
        "source_weight", "language_weight",
        "chrF2", "BLEU"
    ])

    for st_coef, min_prob, temp, src_w, lang_w in itertools.product(
        st_coefs, student_min_probs, student_temperatures, source_weights, language_weights
    ):
        print(f"\n🔧 Running: st_coef={st_coef}, temp={temp}, sw={src_w}, lw={lang_w}")
        out_file = os.path.join(results_dir, out_file_name)

        # Run generation
        run_cmd = f"""python -m scripts.run \
            --model_path {model_path} \
            --language_pairs {language_pair} \
            --source_contrastive 2 \
            --source_weight {src_w} \
            --language_contrastive en ps \
            --language_weight {lang_w} \
            --model_contrastive True \
            --st_coef {st_coef} \
            --student_min_prob {min_prob} \
            --student_temperature {temp}"""
        subprocess.run(run_cmd, shell=True, check=True)

        # Evaluate
        try:
            chrf_raw = subprocess.check_output(
                f"sacrebleu {ref_file} < {out_file} --metrics chrf", shell=True, text=True)
            bleu_raw = subprocess.check_output(
                f"sacrebleu {ref_file} < {out_file} --tokenize flores101", shell=True, text=True)
        
            # Extract scores from plain output
            chrf_score = float(chrf_raw.strip().split()[1])
            bleu_score = float(bleu_raw.strip().split()[2])  # assumes: BLEU = 4.5
        except Exception as e:
            print(f"❌ Evaluation failed: {e}")
            chrf_score = bleu_score = -1

        writer.writerow([st_coef, min_prob, temp, src_w, lang_w, chrf_score, bleu_score])
        f.flush()
        print(f"✅ chrF2: {chrf_score:.2f}, BLEU: {bleu_score:.2f}")
