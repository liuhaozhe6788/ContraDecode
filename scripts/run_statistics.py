import pandas as pd
import sys, os
import argparse
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from data_preparation.preprocessing import get_lang_ident, top_count_4grams_nltk
from sacrebleu.metrics import CHRF
from scripts.run_results import lang_pairs, model_types
from tqdm import tqdm


# tqdm.pandas()
chrf = CHRF()

# --- Core Logic ---
def process_all(root_dir):
    off_target = {m: {"EN": 0, "src": 0} for m in model_types}
    chrf10 = {m: {"HLMT": [], "X-branch": [], "Zulu-related": [], "high-res": [], "all": []} for m in model_types}
    osc = {m: {"HLMT": [], "X-branch": [], "Zulu-related": [], "high-res": [], "all": []} for m in model_types}
    data_size = {"HLMT": 0, "X-branch": 0, "Zulu-related": 0, "high-res": 0, "all": 0}

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
                chrf_vals.append(chrf_val < 10)

                osc_vals.append(top_count_4grams_nltk(h) > top_count_4grams_nltk(r) + 2)

                # Off-target classification
                lang = get_lang_ident(h)
                if lang == "en" and tgt != "en":
                    off_target[model]["EN"] += 1
                elif not ((lang== "fr" and tgt == "ca") or (lang== "ca" and tgt == "fr")) and (lang == src and tgt != src):
                    off_target[model]["src"] += 1

            chrf_freq = sum(chrf_vals)
            osc_freq = sum(osc_vals)

            if group:
                chrf10[model][group].append(chrf_freq)
                osc[model][group].append(osc_freq)
                data_size[group]+=len(chrf_vals)

            if src == "zu" or tgt == "zu":
                chrf10[model]["Zulu-related"].append(chrf_freq)
                osc[model]["Zulu-related"].append(osc_freq)
                data_size["Zulu-related"]+=len(chrf_vals)

            chrf10[model]["all"].append(chrf_freq)
            osc[model]["all"].append(osc_freq)
            data_size["all"]+=len(chrf_vals)

    return off_target, chrf10, osc, data_size

# --- Output generation ---
def export_tables(off_target, chrf10, osc, data_size):
    df_off_target = pd.DataFrame.from_dict(off_target, orient='index').reset_index().rename(columns={"index": "model"})
    
    def agg_group_metric(metric_dict, data_size):
        return pd.DataFrame([
            {
                "model": m,
                "HLMT": sum(v["HLMT"]) / data_size["HLMT"] if v["HLMT"] else 0,
                "X-branch": sum(v["X-branch"]) / data_size["X-branch"] if v["X-branch"] else 0,
                "high-res": sum(v["high-res"]) / data_size["high-res"] if v["high-res"] else 0,
                "Zulu-related": sum(v["Zulu-related"]) / data_size["Zulu-related"] if v["Zulu-related"] else 0,
                "all": sum(v["all"]) / data_size["all"] if v["all"] else 0,
            } for m, v in metric_dict.items()
        ])

    df_chrf10 = agg_group_metric(chrf10, data_size)
    df_osc = agg_group_metric(osc, data_size)

    df_off_target.to_csv("table_off_target.csv", index=False)
    df_chrf10.to_csv("table_chrf_less_10.csv", index=False)
    df_osc.to_csv("table_oscillation.csv", index=False)

    return df_off_target, df_chrf10, df_osc

# --- Entrypoint ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root_dir", type=str, default="out/flores/small100")
    args = parser.parse_args()

    off_target, chrf10, osc, data_size = process_all(args.root_dir)
    df_off, df_chrf, df_osc = export_tables(off_target, chrf10, osc, data_size)
    print(df_off)
    print(df_chrf)
    print(df_osc)