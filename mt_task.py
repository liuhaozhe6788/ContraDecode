import pandas as pd
import logging
import subprocess
import tempfile
import random
import copy
from pathlib import Path
from scripts.utils_run import FLORES101_CONVERT
from sacrebleu import get_source_file
from datasets import load_dataset
from tqdm import tqdm
import os

class MTTask:

    def __init__(self,
                 src_lang: str,
                 tgt_lang: str,
                 testset: str,
                 use_customize: bool,
                 custom_dataset_name: str,
                 model_path: str
                 ):
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang
        self.language_pair = f"{src_lang}-{tgt_lang}"
        self.testset = testset
        self.use_customize = use_customize
        self.custom_dataset_name = custom_dataset_name
        if True:
            base_out_dir = Path("/content/drive/MyDrive/csnlp/out")
        else:
            base_out_dir = Path(__file__).parent / "out"

        print(base_out_dir)
        if not base_out_dir.exists():
            base_out_dir.mkdir(exist_ok=True)
        self.out_dir = base_out_dir / self.testset
        self.out_dir.mkdir(exist_ok=True)
        if model_path.startswith("small100"):
            self.out_dir = self.out_dir / "small100"
            self.out_dir.mkdir(exist_ok=True) 
        elif model_path.startswith("m2m100"):  
            self.out_dir = self.out_dir / "m2m100"
            self.out_dir.mkdir(exist_ok=True)  
        else:
            raise NotImplementedError    

        self.out_dir = self.out_dir / self.language_pair
        self.out_dir.mkdir(exist_ok=True)
           
        self.load_converter = FLORES101_CONVERT

    def __str__(self):
        return f"{self.testset}-{self.src_lang}-{self.tgt_lang}"

    def evaluate(self, translation_method: callable, type='direct', source_contrastive=1, source_weight=None, language_contrastive=None, language_weight=None, student_coef: float = 0.5, student_min_prob: float = 0, student_temperature: float = 0.5, student_alpha=0, use_dynamic_coef=True) -> Path:

        if self.use_customize:
            df = pd.read_csv(f'customized_datasets/{self.custom_dataset_name}')
            source_sentences = df['sentence'].tolist()
        else:
        ## load FLORES dataset
            source_sentences = load_dataset('gsarti/flores_101',self.load_converter[self.src_lang])['devtest']['sentence']

        if type == 'direct':
            translations = translation_method(
            src_lang=self.src_lang,
            tgt_lang=self.tgt_lang,
            source_sentences=source_sentences,
            )
        elif type == 'input_contrastive':
            multi_source_sentences = [source_sentences]
            src_weights = [1]
            tgt_langs=[self.tgt_lang]
            src_langs=[self.src_lang]

            # randomly shuffled input to suppress hallucinations
            if source_contrastive:
                for i in range(source_contrastive):
                    shuffled_sentences = copy.copy(source_sentences)
                    random.shuffle(shuffled_sentences)
                    multi_source_sentences.append(shuffled_sentences)
                    src_weights.append(source_weight/source_contrastive)
                    tgt_langs.append(self.tgt_lang)
                    src_langs.append(self.src_lang)

            # input with wrong target language indicator to suppress off-target translation
            if language_contrastive:
                for offtarget in language_contrastive:
                    # ignore contrastive variants that are identical to true translation direction
                    if offtarget == self.tgt_lang:
                        continue
                    # don't create contrastive variant for src language if language is already listed (avoid duplicates)
                    if offtarget == 'src' and self.src_lang in language_contrastive:
                        continue
                    multi_source_sentences.append(source_sentences)
                    src_weights.append(language_weight)
                    if offtarget == 'src':
                        tgt_langs.append(self.src_lang)
                    else:
                        tgt_langs.append(offtarget)
                    src_langs.append(self.src_lang)
            # now we get multi_source_sentences, including the original source sentence, contrastive source sentences,
            # and the duplicates of original source sentence for contrastive languages, src_weights, src_langs, and tgt_langs
            
            translations = []
            for pair in tqdm(list(zip(*multi_source_sentences))):
                translation = translation_method(
                    src_langs=src_langs,
                    tgt_langs=tgt_langs,
                    src_weights=src_weights,
                    multi_source_sentences=pair,
                    )
                translations.append(translation)
        elif type.startswith('model_contrastive'):  # for teacher-student contrastive decoding
            translations = translation_method(
            src_lang=self.src_lang,
            tgt_lang=self.tgt_lang,
            source_sentences=source_sentences,
            student_coef=student_coef, 
            student_min_prob=student_min_prob, 
            student_temperature=student_temperature,
            student_alpha=student_alpha,
            use_dynamic_coef=use_dynamic_coef
            )
        elif type.startswith('hybrid_contrastive'):
            multi_source_sentences = [source_sentences]
            src_weights = [1]
            tgt_langs=[self.tgt_lang]
            src_langs=[self.src_lang]

            # randomly shuffled input to suppress hallucinations
            if source_contrastive:
                for i in range(source_contrastive):
                    shuffled_sentences = copy.copy(source_sentences)
                    random.shuffle(shuffled_sentences)
                    multi_source_sentences.append(shuffled_sentences)
                    src_weights.append(source_weight/source_contrastive)
                    tgt_langs.append(self.tgt_lang)
                    src_langs.append(self.src_lang)

            # input with wrong target language indicator to suppress off-target translation
            if language_contrastive:
                for offtarget in language_contrastive:
                    # ignore contrastive variants that are identical to true translation direction
                    if offtarget == self.tgt_lang:
                        continue
                    # don't create contrastive variant for src language if language is already listed (avoid duplicates)
                    if offtarget == 'src' and self.src_lang in language_contrastive:
                        continue
                    multi_source_sentences.append(source_sentences)
                    src_weights.append(language_weight)
                    if offtarget == 'src':
                        tgt_langs.append(self.src_lang)
                    else:
                        tgt_langs.append(offtarget)
                    src_langs.append(self.src_lang)
            # now we get multi_source_sentences, including the original source sentence, contrastive source sentences,
            # and the duplicates of original source sentence for contrastive languages, src_weights, src_langs, and tgt_langs
            
            translations = []
            for pair in tqdm(list(zip(*multi_source_sentences))):
                translation = translation_method(
                    src_langs=src_langs,
                    tgt_langs=tgt_langs,
                    src_weights=src_weights,
                    multi_source_sentences=pair,
                    student_coef=student_coef, 
                    student_min_prob=student_min_prob, 
                    student_temperature=student_temperature,
                    student_alpha=student_alpha,
                    use_dynamic_coef=use_dynamic_coef
                    )
                translations.append(translation)          
        else:
            raise NotImplementedError

        # if type == 'direct':
        #     file_name = 'direct'
        # elif type == 'input_contrastive':
        #     file_name = 'input-contrastive-{0}-{1}'.format(source_contrastive, source_weight)
        #     if language_contrastive:
        #         file_name += "-lang-{0}-{1}".format('+'.join(language_contrastive), language_weight)
        # elif type == 'model_contrastive':
        #     file_name = 'model-contrastive'
        # elif type == 'hybrid_contrastive':
        #     file_name = 'hybrid-contrastive'
        # else:
        #     raise NotImplementedError

        out_file = str(self.out_dir)+"/"+type+".txt"
        with open(out_file, 'w') as f:
            f.write("\n".join(translations))

        if not os.path.isfile(str(self.out_dir)+"/"+"ref.text"):
            if self.use_customize:
                ref_file = str(self.out_dir) + "/" + "customized_ref.txt"
                target_sentences = df['reference'].tolist()
                with open(ref_file, 'w') as f:
                    f.write("\n".join(target_sentences))
            else:
                ref_file = str(self.out_dir) + "/" + "ref.txt"
                target_sentences = load_dataset('gsarti/flores_101', self.load_converter[self.tgt_lang])['devtest']['sentence']
                with open(ref_file, 'w') as f:
                    f.write("\n".join(target_sentences))
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
        eval_file = str(self.out_dir)+f"/eval_{type}.txt"
        with open(eval_file, 'w') as f:
            f.write("\n".join([str(chrf_score), str(bleu_score)]))       
        return Path(out_file)
