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


# CONFIG
lang_pairs = [
    # Non-English
    ('af', 'zu'), ('ar', 'fr'), ('be', 'ru'), ('cs', 'sk'), ('de', 'hr'), ('de', 'hu'), ('el', 'tr'),
    ('fr', 'sw'), ('hi', 'bn'), ('hi', 'mr'), ('hr', 'cs'), ('hr', 'hu'), ('hr', 'sk'), ('hr', 'sr'),
    ('it', 'de'), ('it', 'fr'), ('nl', 'de'), ('nl', 'fr'), ('ro', 'de'), ('ro', 'hu'), ('ro', 'hy'),
    ('ro', 'ru'), ('ro', 'tr'), ('ro', 'uk'), ('uk', 'ru'),

    # Low resource
    ("af", "ast"), ("af", "hr"), ("af", "ps"), ("af", "ur"), ("af", "zu"),
    ("ast", "af"), ("ast", "hr"), ("ast", "ps"), ("ast", "ur"), ("ast", "zu"),
    ("hr", "af"), ("hr", "ast"), ("hr", "ps"), ("hr", "ur"), ("hr", "zu"),
    ("ps", "af"), ("ps", "ast"), ("ps", "hr"), ("ps", "ur"), ("ps", "zu"),
    ("ur", "af"), ("ur", "ast"), ("ur", "hr"), ("ur", "ps"), ("ur", "zu"),
    ("zu", "af"), ("zu", "ast"), ("zu", "hr"), ("zu", "ps"), ("zu", "ur"),

    # Added English, French, German pairs (no identity pairs)
    ('en', 'fr'),
    ('fr', 'en'),
    ('en', 'de'),
    ('de', 'en')
]

drop_columns = ['has_image', 'has_hyperlink', 'URL', 'domain', 'topic']

# tqdm.pandas()
chrf = CHRF()
bleu = BLEU(effective_order=True)

lang_ident_model_path = hf_hub_download(repo_id="laurievb/OpenLID", filename="model.bin")
lang_ident_model = fasttext.load_model(lang_ident_model_path)

# get language identification of the sentence
def get_lang_ident(sentence):
    sentence = str(sentence)
    lang_ident = lang_ident_model.predict(sentence)[0][0].split("__")[2].split("_")[0][:2]
    return lang_ident

# get top 4-gram count of the sentence
def top_count_4grams_nltk(sentence):
    tokens = word_tokenize(sentence)
    four_grams = list(ngrams(tokens, 4))
    if not four_grams:
        return 0
    return Counter(four_grams).most_common(1)[0][1]

# Example usage
sentence = "the quick brown fox jumps over the lazy dog"
count = top_count_4grams_nltk(sentence)

# --- Load FLORES dataset ---
def load_language_df(lang_code):
    dataset = load_dataset("gsarti/flores_101", FLORES101_CONVERT[lang_code])
    df = pd.DataFrame(dataset['dev'])
    return df.drop(columns=drop_columns)

# --- Full pipeline for one language pair ---
def process_lang_pair(src, tgt, batch_size):
    print(f"\n Translating from {src} → {tgt}")
    df_src = load_language_df(src)
    df_tgt = load_language_df(tgt)
    
    inputs = df_src['sentence'].to_list()
    translations = model.translate(tgt, inputs, src, batch_size=batch_size)
    
    df_src['translation'] = translations
    df_src['reference'] = df_tgt['sentence']
    
    df_src['chrf2_score'] = [
        chrf.sentence_score(h, [r]).score for h, r in zip(df_src['translation'], df_tgt['sentence'])
    ]
    df_src['is_oscillate'] = [
        top_count_4grams_nltk(h) > top_count_4grams_nltk(r) + 2 for h, r in zip(df_src['translation'], df_tgt['sentence'])
    ]
    df_src['is_not_targ_lang'] = [
        get_lang_ident(h) != tgt for h in df_src['translation']
    ]
    df_src['spbleu_score'] = [
        bleu.sentence_score(h, [r]).score for h, r in zip(df_src['translation'], df_tgt['sentence'])
    ]
    df_src = df_src[(df_src['chrf2_score']< 45.6) | (df_src['spbleu_score']< 18.7) | df_src['is_oscillate'] | df_src['is_not_targ_lang']]
    df_src = df_src[~(df_src['translation'] == df_src['reference'])]

    return df_src[['sentence', 'reference', 'translation']]

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True, choices=["small100", "m2m100_418M"],
                        help="The HF model path")
    parser.add_argument("--batch_size", type=int, default=16,
                        help="The batch size")
    args = parser.parse_args()
    model_path = args.model_path
    batch_size = args.batch_size
    # Load model
    model = load_translation_model(model_path, device=0)
    # --- Process all pairs ---
    writer = pd.ExcelWriter(f'dataset-{model_path}.xlsx', engine='xlsxwriter')
    for src, tgt in lang_pairs:
        df = process_lang_pair(src, tgt, batch_size)
        df.to_excel(writer, sheet_name=f"{src}-{tgt}", index=False)  
    writer.close()