from datasets import load_dataset
import pandas as pd
import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from transformers.models.m2m_100 import M2M100ForConditionalGeneration, M2M100Tokenizer
from tqdm.notebook import tqdm
from mt_task import MTTask
from translation_models import load_translation_model
from sacrebleu.metrics import CHRF, BLEU

from scripts.utils_run import FLORES101_CONVERT
import nltk
from nltk.util import ngrams
from collections import Counter

nltk.download('punkt_tab')
from nltk.tokenize import word_tokenize
import fasttext
from huggingface_hub import hf_hub_download
import argparse
from io import BytesIO
import requests
from tqdm import tqdm

URL = "https://docs.google.com/spreadsheets/d/16Ot7HFcgNNTJPT2pyH2B_fEyfSxJABeV/export?format=xlsx"
response = requests.get(URL)
response.raise_for_status()
xls_data = BytesIO(response.content)

# CONFIG
lang_pairs = {
    "HLMT": [('af', 'zu'), ('ar', 'fr'), ('be', 'ru'), ('cs', 'sk'), ('de', 'hr'), ('de', 'hu'), ('el', 'tr'),
             ('fr', 'sw'), ('hi', 'bn'), ('hi', 'mr'), ('hr', 'cs'), ('hr', 'hu'), ('hr', 'sk'), ('hr', 'sr'),
             ('it', 'de'), ('it', 'fr'), ('nl', 'de'), ('nl', 'fr'), ('ro', 'de'), ('ro', 'hu'), ('ro', 'hy'),
             ('ro', 'ru'), ('ro', 'tr'), ('ro', 'uk'), ('uk', 'ru')],
    "X-branch": [("af", "ast"), ("af", "hr"), ("af", "ps"), ("af", "ur"), ("af", "zu"),
                 ("ast", "af"), ("ast", "hr"), ("ast", "ps"), ("ast", "ur"), ("ast", "zu"),
                 ("hr", "af"), ("hr", "ast"), ("hr", "ps"), ("hr", "ur"), ("hr", "zu"),
                 ("ps", "af"), ("ps", "ast"), ("ps", "hr"), ("ps", "ur"), ("ps", "zu"),
                 ("ur", "af"), ("ur", "ast"), ("ur", "hr"), ("ur", "ps"), ("ur", "zu"),
                 ("zu", "af"), ("zu", "ast"), ("zu", "hr"), ("zu", "ps"), ("zu", "ur")],
    "high-res": [('en', 'fr'), ('fr', 'en'), ('en', 'de'), ('de', 'en')]
}

drop_columns = ['has_image', 'has_hyperlink', 'URL', 'domain', 'topic']

DECODE_METHODS = {
    'direct.txt': 'baseline',
    'input_contrastive.txt': 'Source+lang contrastive (s-contra)',
    'model_contrastive_decoder_only.txt': 'Decoder only stu model',
    'model_contrastive_attention_scaling.txt': 'Attention scaling stu model',
    'hybrid_contrastive_decoder_only.txt': 'S-contra+decoder only',
    'hybrid_contrastive_attention_scaling.txt': 'S-contra+attention scaling',
}

model_types = ["direct", "input_contrastive", "model_contrastive_decoder_only",
               "model_contrastive_attention_scaling", "hybrid_contrastive_decoder_only",
               "hybrid_contrastive_attention_scaling"]

# tqdm.pandas()
chrf = CHRF()
bleu = BLEU(effective_order=True)

lang_ident_model_path = hf_hub_download(repo_id="laurievb/OpenLID", filename="model.bin")
lang_ident_model = fasttext.load_model(lang_ident_model_path)

# get language identification of the sentence
def get_lang_ident(sentence):
    sentence = str(sentence)
    # print(sentence)
    lang_ident = lang_ident_model.predict(sentence)[0][0].split("__")[2].split("_")[0][:2]
    return lang_ident

# get top 4-gram count of the sentence
def top_count_4grams_nltk(sentence):
    tokens = word_tokenize(sentence)
    four_grams = list(ngrams(tokens, 4))
    if not four_grams:
        return 0
    return Counter(four_grams).most_common(1)[0][1]

def load_language_df(lang_code):
    dataset = load_dataset("gsarti/flores_101", FLORES101_CONVERT[lang_code])
    df = pd.DataFrame(dataset['dev'])
    return df.drop(columns=drop_columns)

# --- Core Logic ---
def process_all(root_dir):
    off_target = {m: {"EN": 0, "src": 0} for m in model_types}
    chrf10 = {m: {"HLMT": [], "X-branch": [], "high-res": [], "all": []} for m in model_types}
    osc = {m: {"HLMT": [], "X-branch": [], "high-res": [], "all": []} for m in model_types}

    group_map = {}
    for g, pairs in lang_pairs.items():
        for src, tgt in pairs:
            group_map[f"{src}-{tgt}"] = g

    for lang_pair in tqdm(os.listdir(root_dir), desc="Processing language pairs"):
        if "-" not in lang_pair:
            continue
        src, tgt = lang_pair.split("-")
        lang_tag = f"{src}-{tgt}"
        group = group_map.get(lang_tag, None)

        try:
            # df_src = load_language_df(src)
            # df_tgt = load_language_df(tgt)
            df_tgt = pd.read_excel(xls_data, sheet_name=lang_pair)
            reference = df_tgt["sentence"].tolist()
            # print('REFERENCE')
            # for re in reference[:5]:
            #     print(re)
            # print('------------------------------------------------------------------------------------------------------------------------------------------------')
        except:
            print(f"Skipping {lang_tag} due to loading issue")
            continue

        pair_dir = os.path.join(root_dir, lang_pair)
        for model in model_types:
            file_path = os.path.join(pair_dir, f"{model}.txt")
            if not os.path.exists(file_path):
                continue
            with open(file_path, "r", encoding="utf-8") as f:
                translations = [line.strip() for line in f.readlines()]

            if len(translations) != len(reference):
                # print(f"Skipping {lang_tag} for {model}: length mismatch")
                print('------------------------------------------------------------------------------------')
                print(f"Pair: {lang_tag}, Model:{model}, Translations: {len(translations)}, Reference: {len(reference)}")
                print('------------------------------------------------------------------------------------')
                for tr in translations[:5]:
                    print(tr)
                print('------------------------------------------------------------------------------------------------------------------------------------------------')
                continue

            chrf_vals, osc_vals = [], []
            for h, r in zip(translations, reference):
                chrf_val = chrf.sentence_score(h, [r]).score
                chrf_vals.append(chrf_val < 10)

                osc_vals.append(top_count_4grams_nltk(h) > top_count_4grams_nltk(r) + 2)

                # Off-target classification
                lang = get_lang_ident(h)
                if lang == "en":
                    off_target[model]["EN"] += 1
                elif lang == src:
                    off_target[model]["src"] += 1

            chrf_ratio = sum(chrf_vals) / len(chrf_vals)
            osc_ratio = sum(osc_vals) / len(osc_vals)

            if group:
                chrf10[model][group].append(chrf_ratio)
                osc[model][group].append(osc_ratio)

            chrf10[model]["all"].append(chrf_ratio)
            osc[model]["all"].append(osc_ratio)

    return off_target, chrf10, osc

# --- Output generation ---
def export_tables(off_target, chrf10, osc):
    df_off_target = pd.DataFrame.from_dict(off_target, orient='index').reset_index().rename(columns={"index": "model"})
    
    def agg_group_metric(metric_dict):
        return pd.DataFrame([
            {
                "model": m,
                "HLMT": sum(v["HLMT"]) / len(v["HLMT"]) if v["HLMT"] else 0,
                "X-branch": sum(v["X-branch"]) / len(v["X-branch"]) if v["X-branch"] else 0,
                "high-res": sum(v["high-res"]) / len(v["high-res"]) if v["high-res"] else 0,
                "all": sum(v["all"]) / len(v["all"]) if v["all"] else 0,
            } for m, v in metric_dict.items()
        ])

    df_chrf10 = agg_group_metric(chrf10)
    df_osc = agg_group_metric(osc)

    df_off_target.to_csv("table_off_target.csv", index=False)
    df_chrf10.to_csv("table_chrf_less_10.csv", index=False)
    df_osc.to_csv("table_oscillation.csv", index=False)

    return df_off_target, df_chrf10, df_osc

# --- Entrypoint ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root_dir", type=str, default="out_complete/flores/small100")
    args = parser.parse_args()

    off_target, chrf10, osc = process_all(args.root_dir)
    df_off, df_chrf, df_osc = export_tables(off_target, chrf10, osc)
    print(df_off)
    print(df_chrf)
    print(df_osc)