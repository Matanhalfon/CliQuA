# Runs evaluation flow of CliQuA over LLM inference.
# Note, first always generate input data with preporcess_input.py
# Example usage:
# $ python evaluation_flow.py  -input_annotated_samples_file "test_data/input_file.json" \
#  -output_annotated_samples_file "test_data/inference_output_file.json"

import argparse
import dirtyjson
import json
import os
import re
import numpy as np


from annotated_sample import AnnotatedSample
import evaluation_utils as eu
import inference_utils


SPITTERS = ['[...]','...']


def validate_paths(input_annotated_samples_file,
                   output_annotated_samples_file):
    """Validate paths."""

    # Absolute Paths (Recommended)
    input_annotated_samples_file_path = os.path.abspath(
        input_annotated_samples_file)
    output_annotated_samples_file_path = os.path.abspath(
        output_annotated_samples_file)
    
    # Input Validation
    if not os.path.isfile(input_annotated_samples_file_path):
        raise ValueError(f"Invalid file path: {input_annotated_samples_file}")
    if os.path.exists(output_annotated_samples_file_path):
        raise ValueError(f"File already exists: {output_annotated_samples_file}")

def generate_prompt(annotated_sample: AnnotatedSample):
    prompt = f"""# Task Your job is to decide whether the given patient meets the inclusion criterion.

    # Patient
    Below is a clinical note describing the patient's current health status: ```{annotated_sample.annotated_passages[0].passage}```

    # Inclusion Criterion 
    The inclusion criterion being assessed is: "{annotated_sample.query}" 

    # Assessment 
    Given the criterion above, use the patient's clinical note to determine whether the patient meets this criterion. Think step by step, and justify your answer. 

    Format your response as a JSON object with the following keys: 
    * rationale: str - Your reasoning as to why the patient does or does not meet the criterion
    * quote: str - a quote from the patient's clinical note that supports your rationale
    * is_met: bool - "true" if the patient meets that criterion, or it can be inferred that they meet that criterion with common sense. "false" if the patient does not or it is impossible to assess this given the provided information. 
    An example of how your JSON response should be formatted is shown below: 
    ```json {{ "rationale" : "something something", "quote" : "short string from patients’ clinical notes" "is_met" : true/false }} ```
    The above example is only for illustration purposes only.
    It does not reflect the actual criterion or patient for this task. 


    Please provide your response:"""
    return prompt


def run_llm(prompts: list[str], model: inference_utils.LLMInference) -> list[str]:
    return model.inference(prompts)


def normalize_raw_answer(raw_answer: str) -> str:
    # json_string = '''```json
    # {
    #   "rationale": "bla bla bla,
    #   "quote": "NORVASC	10MG PO QD for her CAD.",
    #   "is_met": true
    # }
    # ```'''
    json_string  = raw_answer.strip("\n").strip().strip("`").strip('json').strip("\n").strip("`").strip("\n")
    if '```' in json_string:
        json_string = json_string.split('```')[0]

    # Handle cases of newlines.
    json_string = json_string.replace('\\n" + \n"', '\\n')
    json_string=json_string.replace("\n\n", "\n")
    json_string=json_string.replace(".\n", ". ")
    json_string = re.sub(r'(?<=[a-zA-Z])\n(?=[a-zA-Z])', ' ', json_string)
  
    # Handle issues with in-text quotations.
    json_string=json_string.replace('''"quote" : “''', '''"quote" : "''')
    json_string=json_string.replace('''".", "quote"''', '''“.", "quote"''')
  
    # Handle cases where a comma is present after last entry.
    json_string=json_string.replace(",\n}", "\n}")
    json_string = json_string.replace('"is_met": True', '"is_met": true')
    json_string = json_string.replace('"is_met": False', '"is_met": false')
    json_string = json_string.replace('''Has the patient experienced angina?"''',
                                      '''Has the patient experienced angina?\'''') 
    
    # Fix chat mode concat.
    json_string = json_string.replace('"quote\n"', '"quote"')
    json_string = json_string.replace('"\nquote"', '"quote"')
    json_string = json_string.replace('"is_met\n"', '"is_met"')
    json_string = json_string.replace('"\nis_met"', '"is_met"')
    json_string = json_string.replace('"is\n_met"', '"is_met"')
    json_string = json_string.replace('"is_\nmet"', '"is_met"')

    json_string = json_string.replace("\\\"","'")
    json_string = json_string.replace('"""','"')
    json_string.strip("\n")
    return json_string


def parse_llm_answers(raw_answers: list[str],
                      annotated_samples: list[AnnotatedSample]) -> list[AnnotatedSample]:

    gt_positive_samples = 0
    total_samples = 0
    json_failed = 0
    if len(raw_answers) != len(annotated_samples):
        raise ValueError(
            f"Number of inferences and annotated samples don't match")

    for i in range(len(annotated_samples)):
        raw_answer = raw_answers[i]
        annotated_samples[i].annotated_passages[0].model_annotation.answer = raw_answer
        total_samples += 1    
        json_string = normalize_raw_answer(raw_answer) 
        try:
            parsed_answer = dirtyjson.loads(json_string)
            annotated_samples[i].annotated_passages[0].model_annotation.label = parsed_answer['is_met']
            if type(annotated_samples[i].annotated_passages[0].model_annotation.label) == str:
                annotated_samples[i].annotated_passages[0].model_annotation.label = True if parsed_answer['is_met'].lower() in ['true','yes','match' ] else False
            if parsed_answer['quote']:
                annotated_samples[i].annotated_passages[0].model_annotation.reference = parsed_answer['quote'].strip('"')
            if parsed_answer['rationale']:
                annotated_samples[i].annotated_passages[0].model_annotation.explanation = parsed_answer['rationale'].strip('"')       
        except:
            json_failed +=1
            json_string = raw_answer
            annotated_samples[i].annotated_passages[0].model_annotation.reference = None
            annotated_samples[i].annotated_passages[0].model_annotation.label = False
            
            #GPT4 & GEMINI: 
            quote_match = re.search('''"quote": "(.*)"''', json_string)

            #Bison32:
            if not quote_match or not quote_match.group(1):
                quote_match = re.search('''"quote"\s*:\s*"(.*)''', json_string.strip())
                
            met_match = re.search('''"is_met": (.*)\n''',json_string)
            if quote_match:
                annotated_samples[i].annotated_passages[0].model_annotation.reference = quote_match.group(1)
            if met_match:
                annotated_samples[i].annotated_passages[0].model_annotation.label = True if met_match.group(1).lower() == 'true' else False
            else:
                if quote_match and quote_match.group(1):
                    annotated_samples[i].annotated_passages[0].model_annotation.label = True
                              
                
        if annotated_samples[i].ground_truth_label == True:
            gt_positive_samples +=1
    print("Failed to parse JSON: ",json_failed)
    print("# samples: ", total_samples)
    return annotated_samples

def get_positives_only(model_annotation_data_parsed):
    pos_only = []
    for sample in model_annotation_data_parsed:
        if sample.ground_truth_label:
            pos_only.append(sample)
    return pos_only

def print_match_attribution_metrics(model_annotations, model_name, model_path,
                                    print_header = False):
    
    # Calculate attribution and binary F1 for all samples
    all_stats = eu.calc_stats(model_annotations, require_model_match=False,
                              require_model_attributions=False)
    total = (all_stats.false_negatives+all_stats.true_negatives+
             all_stats.false_positives + all_stats.true_positives)    
    gt_positives = all_stats.true_positives + all_stats.false_negatives
    model_positives = all_stats.false_positives + all_stats.true_positives
    print(gt_positives, model_positives, all_stats.true_positives)
    recall = all_stats.true_positives / gt_positives
    precision = all_stats.true_positives / model_positives
    f1 = 2*recall*precision/(recall + precision)
    f1_attribution_all = all_stats.max_f1

    # Calculate attribution stats for positive samples.
    pos_stats = eu.calc_stats(get_positives_only(model_annotations),
                              require_model_match=True,
                              require_model_attributions=True)
    f1_attribution_pos = pos_stats.max_f1
    # Valid attributions are only calculated for True Positives.
    valid_attribution_rate = pos_stats.tp_attributions / pos_stats.true_positives
    # Note avg positive attribution len does not include empty/missing attributions.
    att_len = eu.avg_positive_attribution_len(model_annotations)
    
    # Calculate attribution groundness.
    (pos_ungrounded, pos_empty, neg_ungrounded, neg_empty) = eu.attribution_grounding_stats(
        model_annotations)   
    (pos_ungrounded_w_split, pos_empty_w_split, neg_ungrounded_w_split,
      neg_empty_w_split) = eu.attribution_grounding_stats(
        model_annotations, split_indicators=SPITTERS)

    if print_header:
        print(f"Model Name, Input, Total Samples, Match Recall,"
              f"Match Precision,Match F1, F1 Attribution All,"
              f"F1 Attribution Positive, Valid Attribution Rate (TP only),"
              f"TP, FP, TN, FN, Positive Attribution Length,",
              f"Positive ungrounded, Positive Empty, Negative ungrounded,"
              f"Negative empty,"
              f"Positive ungrounded w splits, Positive Empty  w splits,"
              f"Negative ungrounded  w splits, Negative empty w splits,"
              f"Grounded ratio, Grounded w split ratio")

    print(f"{model_name}, {model_path}, {total}, {recall}, {precision}, {f1},"
          f"{f1_attribution_all}, {f1_attribution_pos},"
          f"{valid_attribution_rate}, {all_stats.true_positives},"
          f"{all_stats.false_positives}, {all_stats.true_negatives},"
          f"{all_stats.false_negatives}, {att_len},",
          f"{pos_ungrounded}, {pos_empty}, {neg_ungrounded}, {neg_empty},"
          f"{pos_ungrounded_w_split}, {pos_empty_w_split}, {neg_ungrounded_w_split}, {neg_empty_w_split},"
          f"{1-(pos_ungrounded+pos_empty)/model_positives},"
          f"{1-(pos_ungrounded_w_split+pos_empty_w_split)/model_positives}")


def main(input_annotated_samples_file:str, output_annotated_samples_file:str
         ):
    # Read annotated samples.
    with open(input_annotated_samples_file, 'r') as f:
        input_json = json.load(f)
        annotated_samples = []
        for json_sample in input_json['data']:
            annotated_samples.append(AnnotatedSample.from_dict(json_sample))
    
    # Generate prompts.
    prompts = [generate_prompt(sample) for sample in annotated_samples]

    # Run LLM over prompts.
    model = inference_utils.HFInference("google/gemma-2b-it")
    raw_answers = run_llm(prompts, model)

    # Parse LLM answers and populate into annotated samples.
    annotaed_samples_with_model = parse_llm_answers(raw_answers, annotated_samples)

    # Write out annotated samples.
    annotated_samples_json = {'data': [
        sample.to_dict() for sample in annotaed_samples_with_model]}
    print(f"Writing joint AnnotatedSamples out to {output_annotated_samples_file}")
    with open(output_annotated_samples_file, 'w') as f:
        json.dump(annotated_samples_json, f)

    # Run evaluation.
    print_match_attribution_metrics(annotaed_samples_with_model, "test","none",True)



if __name__ == "__main__":
    # Argument Parsing
    parser = argparse.ArgumentParser(
        description="Generates prompts and evaluation data from LLMs")
    parser.add_argument(
        "-input_annotated_samples_file",
        help="Path to annotated samples file generated by preprocess_input.py")
    parser.add_argument(
        "-output_annotated_samples_file",
        help="Output file for model annotations.")
    args = parser.parse_args()

    try:
        validate_paths(args.input_annotated_samples_file,
                       args.output_annotated_samples_file)
        main(args.input_annotated_samples_file,
             args.output_annotated_samples_file)
    except ValueError as e:
        print(f"Error: {e}")  