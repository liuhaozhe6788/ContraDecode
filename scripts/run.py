import logging
import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import translation_models
from mt_task import MTTask
from translation_models import load_translation_model
from scripts.utils_run import FLORES101_CONVERT, M2M100_TYPES, IGNORE_FOR_FLORES, FAMILIES
from itertools import combinations
import argparse

logging.basicConfig(level=logging.INFO)

def float_range(mini,maxi):
    """Return function handle of an argument type function for 
       ArgumentParser checking a float range: mini <= arg <= maxi
         mini - minimum acceptable argument
         maxi - maximum acceptable argument"""

    # Define the function with default arguments
    def float_range_checker(arg):
        """New Type function for argparse - a float within predefined range."""

        try:
            f = float(arg)
        except ValueError:    
            raise argparse.ArgumentTypeError("must be a floating point number")
        if f < mini or f > maxi:
            raise argparse.ArgumentTypeError("must be in range [" + str(mini) + " .. " + str(maxi)+"]")
        return f

    # Return function handle to checking function
    return float_range_checker

def main(args):

    model = load_translation_model(args.model_path, device=0)
    language_pairs = args.language_pairs.split(',')
    use_customized = args.use_customized
    custom_dataset_name = args.customized_dataset_name
    language_pairs = [x.split("-") for x in language_pairs]

    if args.oneshot:
        assert isinstance(model, translation_models.llama.LLaMaTranslationModel)
        model.one_shot = True

    tasks = []
    
    for lang_pair in language_pairs:
        tasks.append(MTTask(lang_pair[0],lang_pair[1],'flores',use_customized,custom_dataset_name))
        print(f"Task added {lang_pair[0]} - {lang_pair[1]}")

    for task in tasks:
        if (args.source_contrastive or args.language_contrastive) and not args.model_contrastive:  
            print(f"Evaluating {task} multi_source")
            out_path = task.evaluate(model.translate_multi_source, 'input_contrastive', args.source_contrastive, args.source_weight, args.language_contrastive, args.language_weight)
            print(f"Translations saved in {out_path}")
        elif not (args.source_contrastive or args.language_contrastive) and args.model_contrastive:  
            print(f"Evaluating {task} teacher-student") 
            out_path = task.evaluate(model.translate_teacher_student, 'model_contrastive', st_coef=args.st_coef, student_min_prob=args.student_min_prob, student_temperature=args.student_temperature)
            print(f"Translations saved in {out_path}")     
        elif (args.source_contrastive or args.language_contrastive) and args.model_contrastive:  
            print(f"Evaluating {task} hybrid") 
            out_path = task.evaluate(model.translate_hybrid, 'hybrid_contrastive', args.source_contrastive, args.source_weight, args.language_contrastive, args.language_weight, args.st_coef, args.student_min_prob, args.student_temperature)
            print(f"Translations saved in {out_path}")                
        else:
            print(f"Evaluating {task} direct")
            out_path = task.evaluate(model.translate, 'direct')
            print(f"Translations saved in {out_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="",
                        help="The HF model path")
    parser.add_argument("--use_customized", action="store_true",
                    help="Use customized dataset instead of Flores")
    parser.add_argument("--customized_dataset_name", type=str, default="customized_dataset.csv",
                    help="Path or URL to the customized dataset (required if --use_customized is set).")
    parser.add_argument("--language_pairs", type=str, default="",
                        help="language pairs")
    parser.add_argument("--source_contrastive", nargs='?', const=1, type=int,
                        help="enable contrastive input (randomly shuffled segments from input file). Optional INT defines number of contrastive inputs (default: 1).")
    parser.add_argument("--source_weight", type=float_range(-1, 0), default=-0.1,
                        help="weight of source-contrastive variant. Default -0.1. If multiple contrastive inputs are used, this defines total weight assigned to them.")
    parser.add_argument("--language_contrastive", type=str, nargs='+', default=None,
                        help="language codes of languages for which to construct contrastive variants. Can be multiple (space-separated); 'src' will be mapped to language code of source language. Example: '--language_contrastive en src' will create two contrastive inputs, one with English, one with the source language as desired output language.")
    parser.add_argument("--language_weight", type=float_range(-1, 0), default=-0.1,
                        help="weight of contrastive variants with wrong language indicator. Default -0.1. If multiple contrastive inputs are used, this specifies weight assigned to each of them individually.")
    parser.add_argument("--model_contrastive", type=bool, default=False,
                        help="enable teacher-student contrastive decoding.")
    parser.add_argument("--st_coef", type=float_range(0, 1), default=0.5,
                        help="weight of contrastive variants for student model.")  
    parser.add_argument("--student_min_prob", type=float_range(0, 1), default=0,
                        help="minimum probability for student model.")    
    parser.add_argument("--student_temperature", type=float_range(0, float('inf')), default=1,
                        help="temperature in model inference for student model."),   
    parser.add_argument("--oneshot", action='store_true', default=False,
                        help="For LLaMa: provide one-shot translation example")
    args = parser.parse_args()
    main(args)
