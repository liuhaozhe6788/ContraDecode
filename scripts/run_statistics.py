import pandas as pd
import sys, os
import argparse
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from data_preparation.preprocessing import top_count_4grams_nltk
from sacrebleu.metrics import CHRF
from scripts.run_autoeval import lang_pairs, model_types
from tqdm import tqdm
import fasttext
from huggingface_hub import hf_hub_download

chrf = CHRF()

lang_ident_model_path = hf_hub_download(repo_id="laurievb/OpenLID-v2", filename="model.bin")
lang_ident_model = fasttext.load_model(lang_ident_model_path)

# get language identification of the sentence
def get_lang_ident(sentence):
    sentence = str(sentence)
    lang_ident = lang_ident_model.predict(sentence)[0][0].split("__")[2].split("_")[0][:2]
    return lang_ident

# --- Core Logic ---
def process_all(root_dir):
    off_target = {m: {"EN": 0, "other-src": 0, "low-src": 0} for m in model_types}
    chrf_low = {m: {"HLMT": [], "X-branch": [], "high-res": [], "all": []} for m in model_types}
    osc = {m: {"HLMT": [], "X-branch": [], "high-res": [], "all": []} for m in model_types}
    data_size = {"HLMT": 0, "X-branch": 0, "high-res": 0, "all": 0}

    # map lang-pair to group name
    group_map = {}
    for g, pairs in lang_pairs.items():
        for src, tgt in pairs:
            group_map[f"{src}-{tgt}"] = g

    pairs = []
    for val in list(lang_pairs.values()):
        pairs+=val
    for lang_pair in tqdm(pairs, desc="Processing language pairs"):
        src, tgt = lang_pair
        lang_tag = f"{src}-{tgt}"
        group = group_map.get(lang_tag, None)

        pair_dir = os.path.join(root_dir, lang_tag)

        ref_file_path = os.path.join(pair_dir, f"customized_ref.txt")
        with open(ref_file_path, "r", encoding="utf-8") as f:
            references = [line.strip() for line in f.readlines()]
        for model in model_types:
            file_path = os.path.join(pair_dir, f"{model}.txt")
            with open(file_path, "r", encoding="utf-8") as f:
                translations = [line.strip() for line in f.readlines()]

            chrf_vals, osc_vals = [], []

            assert len(translations) == len(references)
            for h, r in zip(translations, references):
                chrf_val = chrf.sentence_score(h, [r]).score
                chrf_vals.append(chrf_val < 45.6)

                osc_vals.append(top_count_4grams_nltk(h) > top_count_4grams_nltk(r) + 2)

                # Off-target classification
                lang = get_lang_ident(h)
                if lang == "en" and tgt != "en":
                    off_target[model]["EN"] += 1                   
                elif not ((lang== "fr" and tgt == "ca") or (lang== "ca" and tgt == "fr")) and (lang == src and tgt != src):
                    if src in ["af", "ast", "hr", "ps", "ur", "zu"]:
                        off_target[model]["low-src"] += 1 
                    else:
                        off_target[model]["other-src"] += 1

            chrf_freq = sum(chrf_vals)
            osc_freq = sum(osc_vals)

            if group:
                chrf_low[model][group].append(chrf_freq)
                osc[model][group].append(osc_freq)
                data_size[group]+=len(chrf_vals)

            chrf_low[model]["all"].append(chrf_freq)
            osc[model]["all"].append(osc_freq)
            data_size["all"]+=len(chrf_vals)

    return off_target, chrf_low, osc, data_size

# --- Output generation ---
def export_tables(off_target, chrf_low, osc, data_size):
    df_off_target = pd.DataFrame.from_dict(off_target, orient='index').reset_index().rename(columns={"index": "model"})
    
    def agg_group_metric(metric_dict, data_size):
        return pd.DataFrame([
            {
                "model": m,
                "HLMT": sum(v["HLMT"]) / data_size["HLMT"] * 100 if data_size["HLMT"] else 0,
                "X-branch": sum(v["X-branch"]) / data_size["X-branch"] * 100 if data_size["X-branch"] else 0,
                "high-res": sum(v["high-res"]) / data_size["high-res"] * 100 if data_size["high-res"] else 0,
                "all": sum(v["all"]) / data_size["all"] * 100 if data_size["all"] else 0,
            } for m, v in metric_dict.items()
        ])

    df_chrf_low = agg_group_metric(chrf_low, data_size).round(3)
    df_osc = agg_group_metric(osc, data_size).round(3)

    df_off_target.to_csv("table_off_target.csv", index=False)
    df_chrf_low.to_csv("table_chrf_low.csv", index=False)
    df_osc.to_csv("table_oscillation.csv", index=False)

    return df_off_target, df_chrf_low, df_osc

# --- Entrypoint ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root_dir", type=str, default="out/flores/small100")
    args = parser.parse_args()

    off_target, chrf_low, osc, data_size = process_all(args.root_dir)
    df_off, df_chrf, df_osc = export_tables(off_target, chrf_low, osc, data_size)
    print(df_off)
    print(df_chrf)
    print(df_osc)