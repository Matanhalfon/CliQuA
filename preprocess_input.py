# Generate evaluation data for CliQuA evaluation pipeline.
# Example usage:
# $ python preprocess_input.py  -context_dir "test_data/n2c2_context_dir" \
# -ground_truth_file "test_data/ground_truth_cliqua.csv" \
# -output_file "test_data/inference_input_file.json"

import json
import os
import argparse
import xml.etree.ElementTree as ET
import csv
from annotated_sample import AnnotatedSample, AnnotatedPassage


def validate_paths(context_dir: str, ground_truth_file: str,
                   output_file: str):
    """Validate paths."""

    # Absolute Paths (Recommended)
    context_dir_path = os.path.abspath(context_dir)
    ground_truth_file_path = os.path.abspath(ground_truth_file)
    output_file_path = os.path.abspath(output_file)
    
    # Input Validation
    if not os.path.isdir(context_dir_path):
        raise ValueError(f"Invalid directory path: {context_dir}")
    if not os.path.isfile(ground_truth_file_path):
        raise ValueError(f"Invalid file path: {ground_truth_file}")
    if os.path.exists(output_file_path):
        raise ValueError(f"File already exists: {output_file}")

def extract_context(xml_string:str) -> str:
    "Extracts only the patients record from XML."

    root = ET.fromstring(xml_string)

    # Find the <PatientMatching> element first
    for value in root:
        if value.tag != "TEXT":
            continue
        extracted_text = value.text
    if not extracted_text:
        print("TEXT element not found.")
        return

    # Extract and clean text (handle CDATA & remove extra whitespace)
    if extracted_text.startswith("<![CDATA["):
        extracted_text = extracted_text[9:-3]  # Remove CDATA markers

    return extracted_text

def generate_context_per_patient(context_dir:str) -> dict[int, str]:
    patient2records = {}
    for patient_file in os.listdir(context_dir):
        patient_file_path = os.path.join(context_dir, patient_file)  
        if os.path.isfile(patient_file_path):
            print(f"Processing file: {patient_file}")
            with open(patient_file_path, "r") as f:
                full_patient_xml = f.read()
                patient_records = extract_context(full_patient_xml)
                patient2records[patient_file.strip(".xml")] = patient_records
    return patient2records

def read_ground_truth(ground_truth_file:str) -> dict:
    data_dict = {}
    with open(ground_truth_file, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            file_name = row['file_name']
            if file_name not in data_dict:
                data_dict[file_name] =  {}
            label = row['label']
            if label not in data_dict[file_name]:
                data_dict[file_name][label] = []
            attribution = row['text']
            if attribution not in data_dict[file_name][label]:
                data_dict[file_name][label].append(attribution)
                
    return data_dict

def read_labels(labels2questions_file:str) -> dict:
    labes2questions = {}
    with open(labels2questions_file, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            labes2questions[row['label']]=row['question']
    return labes2questions

def main(context_dir:str, ground_truth_file: str, output_file: str):
    # Collect data componenets.
    print(f"Reading all patient records from {context_dir}.")
    patient2records = generate_context_per_patient(context_dir)
    print(f"{len(patient2records)} patients records read.")
    
    print(f"Reading annotations from {context_dir}.")
    patient2annotations = read_ground_truth(ground_truth_file)
    print(f"Read annotations for {len(patient2annotations)} patients.")

    print("Reading labels from label2question.csv")
    labels2questions = read_labels("label2question.csv")
    
    # Combine components into AnnotatedSamples.
    annotated_samples = []
    for label in labels2questions:
        for patient in patient2records:
            ground_truth_attributions = []
            if label in patient2annotations[patient]:
                ground_truth_attributions = patient2annotations[patient][label]
            annotated_passage = AnnotatedPassage(
                0, patient2records[patient],
                ground_truth_attributions, None)
            annotated_sample = AnnotatedSample(
                f"{patient}_{label}",patient, labels2questions[label],
                len(ground_truth_attributions) > 0,
                [annotated_passage])
            annotated_samples.append(annotated_sample)
    
    # Write annotated samples to file. 
    annotated_samples_json = {'data': [
        sample.to_dict() for sample in annotated_samples]}
    print(f"Writing joint AnnotatedSamples out to {output_file}")
    with open(output_file, 'w') as f:
        json.dump(annotated_samples_json, f)



if __name__ == "__main__":
    # Argument Parsing
    parser = argparse.ArgumentParser(
        description="Generates prompts and evaluation data from LLMs")
    parser.add_argument("-context_dir",
                        help="Path to the N2C2 Golden Data Set XMLs")
    parser.add_argument("-ground_truth_file",
                        help="Path to CliQuA annotations file")
    parser.add_argument("-output_file",
                        help="Path to output file.")
    args = parser.parse_args()

    try:
        validate_paths(args.context_dir, args.ground_truth_file,
                        args.output_file)
        main(args.context_dir, args.ground_truth_file, args.output_file)
    except ValueError as e:
        print(f"Error: {e}")
