import os
import argparse
import csv

lang_pairs = {
    "HLMT":
    # Non-English
    [('af', 'zu'), ('ar', 'fr'), ('be', 'ru'), ('cs', 'sk'), ('de', 'hr'), ('de', 'hu'), ('el', 'tr'),
    ('fr', 'sw'), ('hi', 'bn'), ('hi', 'mr'), ('hr', 'cs'), ('hr', 'hu'), ('hr', 'sk'), ('hr', 'sr'),
    ('it', 'de'), ('it', 'fr'), ('nl', 'de'), ('nl', 'fr'), ('ro', 'de'), ('ro', 'hu'), ('ro', 'hy'),
    ('ro', 'ru'), ('ro', 'tr'), ('ro', 'uk'), ('uk', 'ru')],

    # Low resource
    "X-branch":
    [("af", "ast"), ("af", "hr"), ("af", "ps"), ("af", "ur"), ("af", "zu"),
    ("ast", "af"), ("ast", "hr"), ("ast", "ps"), ("ast", "ur"), ("ast", "zu"),
    ("hr", "af"), ("hr", "ast"), ("hr", "ps"), ("hr", "ur"), ("hr", "zu"),
    ("ps", "af"), ("ps", "ast"), ("ps", "hr"), ("ps", "ur"), ("ps", "zu"),
    ("ur", "af"), ("ur", "ast"), ("ur", "hr"), ("ur", "ps"), ("ur", "zu"),
    ("zu", "af"), ("zu", "ast"), ("zu", "hr"), ("zu", "ps"), ("zu", "ur")],

    # Added English, French, German pairs (no identity pairs)
    "high-res":
    [('en', 'fr'),
    ('fr', 'en'),
    ('en', 'de'),
    ('de', 'en')]
}

model_types = ["direct", "input_contrastive", "model_contrastive_decoder_only", "model_contrastive_attention_scaling", "hybrid_contrastive_decoder_only", "hybrid_contrastive_attention_scaling"]

field_names = ["model_type", "chrF2_HLMT", "chrF2_X-branch", "chrF2_high-res", "chrF2_all", "spBLEU_HLMT", "spBLEU_X-branch", "spBLEU_high-res", "spBLEU_all"]

def main(args):
    model_name = args.model_name
    dir = f"out/flores/{model_name}"
    subdirs = None
    for root, dirs, files in os.walk(dir):
        if len(dirs): subdirs = dirs

    # compute average chrf2 and spbleu for HLMT group, X-branch group, high-res group, and all language-pair
    csv_file = f"evaluation_results_{model_name}.csv"
    with open(csv_file, "w", newline="") as f_csv:
        writer = csv.DictWriter(f_csv, fieldnames=field_names)
        new_row = dict(zip(field_names, field_names))
        writer.writerow(new_row)

        chrF2_HLMT = dict()
        chrF2_X_branch = dict()
        chrF2_high_res = dict()
        chrF2_all = dict()

        spbleu_HLMT = dict()
        spbleu_X_branch = dict()
        spbleu_high_res = dict()
        spbleu_all = dict()
        for model_type in model_types:
            chrF2_HLMT[model_type] = []
            chrF2_X_branch[model_type] = []
            chrF2_high_res[model_type] = []
            chrF2_all[model_type] = []

            spbleu_HLMT[model_type] = []
            spbleu_X_branch[model_type] = []
            spbleu_high_res[model_type] = []
            spbleu_all[model_type] = []

        for model_type in model_types:
            for subdir in subdirs:
                sub_dir = os.path.join(dir, subdir)
                f = open(f'{sub_dir}/eval_{model_type}.txt')
                content = f.read()
                f.close()
                chrf2, spbleu = content.split("\n")
                chrf2, spbleu = float(chrf2), float(spbleu)
                chrF2_all[model_type].append(chrf2)
                spbleu_all[model_type].append(spbleu)                

                hlmt_pairs = []
                for pair in lang_pairs["HLMT"]:
                    src, tgt = pair[0], pair[1]
                    hlmt_pairs.append(f"{src}-{tgt}")
                if subdir in hlmt_pairs:
                    chrF2_HLMT[model_type].append(chrf2)
                    spbleu_HLMT[model_type].append(spbleu)   

                x_pairs = []
                for pair in lang_pairs["X-branch"]:
                    src, tgt = pair[0], pair[1]
                    x_pairs.append(f"{src}-{tgt}")
                if subdir in x_pairs:
                    chrF2_X_branch[model_type].append(chrf2)
                    spbleu_X_branch[model_type].append(spbleu)  

                high_res_pairs = []
                for pair in lang_pairs["high-res"]:
                    src, tgt = pair[0], pair[1]
                    high_res_pairs.append(f"{src}-{tgt}")
                if subdir in high_res_pairs:
                    chrF2_high_res[model_type].append(chrf2)
                    spbleu_high_res[model_type].append(spbleu) 
        for model_type in model_types:
            chrF2_HLMT_avg = sum(chrF2_HLMT[model_type])/len(chrF2_HLMT[model_type]) if len(chrF2_HLMT[model_type]) else 0
            chrF2_X_branch_avg = sum(chrF2_X_branch[model_type])/len(chrF2_X_branch[model_type]) if len(chrF2_X_branch[model_type]) else 0
            chrF2_high_res_avg = sum(chrF2_high_res[model_type])/len(chrF2_high_res[model_type]) if len(chrF2_high_res[model_type]) else 0
            chrF2_all_avg = sum(chrF2_all[model_type])/len(chrF2_all[model_type]) if len(chrF2_all[model_type]) else 0

            spbleu_HLMT_avg = sum(spbleu_HLMT[model_type])/len(spbleu_HLMT[model_type]) if len(spbleu_HLMT[model_type]) else 0
            spbleu_X_branch_avg = sum(spbleu_X_branch[model_type])/len(spbleu_X_branch[model_type]) if len(spbleu_X_branch[model_type]) else 0
            spbleu_high_res_avg = sum(spbleu_high_res[model_type])/len(spbleu_high_res[model_type]) if len(spbleu_high_res[model_type]) else 0
            spbleu_all_avg = sum(spbleu_all[model_type])/len(spbleu_all[model_type]) if len(spbleu_all[model_type]) else 0
            new_row = dict(zip(field_names, [model_type, chrF2_HLMT_avg, chrF2_X_branch_avg, chrF2_high_res_avg, chrF2_all_avg, spbleu_HLMT_avg, spbleu_X_branch_avg, spbleu_high_res_avg, spbleu_all_avg]))
            writer.writerow(new_row)
        f_csv.flush()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, required=True, choices=["small100", "m2m100_418M"],
                        help="The translation model name")
    args = parser.parse_args()
    main(args)