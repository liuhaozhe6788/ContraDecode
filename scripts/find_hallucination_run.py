from sacrebleu.metrics import CHRF, BLEU
import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from data_preparation.preprocessing import top_count_4grams_nltk
import argparse

def main(args):
    model_name = args.model_name
    lang_pair = args.lang_pair
    dir = f"out/flores/{model_name}/{lang_pair}"
    chrf = CHRF()

    f = open(f'{dir}/customized_ref.txt')
    references = f.read()
    references = references.split("\n")
    f.close()

    f = open(f'{dir}/direct.txt')
    translations_direct = f.read()
    translations_direct = translations_direct.split("\n")
    f.close()

    f = open(f'{dir}/input_contrastive.txt')
    translations_input_contra = f.read()
    translations_input_contra = translations_input_contra.split("\n")
    f.close()

    f = open(f'{dir}/model_contrastive_attention_scaling.txt')
    translations_model_contrastive_attention_scaling = f.read()
    translations_model_contrastive_attention_scaling = translations_model_contrastive_attention_scaling.split("\n")
    f.close()

    f = open(f'{dir}/model_contrastive_decoder_only.txt')
    translations_model_contrastive_decoder_only = f.read()
    translations_model_contrastive_decoder_only = translations_model_contrastive_decoder_only.split("\n")
    f.close()

    f = open(f'{dir}/hybrid_contrastive_attention_scaling.txt')
    translations_hybrid_contrastive_attention_scaling = f.read()
    translations_hybrid_contrastive_attention_scaling = translations_hybrid_contrastive_attention_scaling.split("\n")
    f.close()

    f = open(f'{dir}/hybrid_contrastive_decoder_only.txt')
    translations_hybrid_contrastive_decoder_only = f.read()
    translations_hybrid_contrastive_decoder_only = translations_hybrid_contrastive_decoder_only.split("\n")
    f.close()

    for i in range(len(references)):
        r, t_direct, t_CD = references[i], translations_direct[i], translations_model_contrastive_decoder_only[i]
        if top_count_4grams_nltk(t_CD) > top_count_4grams_nltk(r) + 2:
            print(i+1)
            print("ref:"+ references[i])
            print("direct:"+ translations_direct[i])
            print("input-CD:"+ translations_input_contra[i])
            print("attn-scale CD:"+ translations_model_contrastive_attention_scaling[i])
            print("decoder-only CD:"+ translations_model_contrastive_decoder_only[i])
            print("joint attn-scale:"+ translations_hybrid_contrastive_attention_scaling[i])
            print("joint decoder-only:"+ translations_hybrid_contrastive_decoder_only[i])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--lang_pair", type=str, required=True,
                        help="The translation language pair")   
    parser.add_argument("--model_name", type=str, required=True, choices=["small100", "m2m100_418M"],
                        help="The translation model name")
    args = parser.parse_args()
    main(args)


