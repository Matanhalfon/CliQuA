# CliQuA: A Benchmark for Clinical Records Question Answering
An eval framework that enables LLM inference over the new dataset release of the paper - "CliQuA: A Benchmark for Clinical Records Question Answering"


## Prerequisites
1. Attain access to N2C2 data set [file](https://portal.dbmi.hms.harvard.edu/projects/n2c2-nlp/).
2. Download the N2C2 dataset under "*2018 (Track 1) Clinical Trial Cohort Selection Challenge Downloads*" download the following repositories:
 - *Test Data: Gold Standard test data.*
 - *Training Data: Gold standard training data.*

This directories will hold test_data/n2c2_context_dir and train_data/n2c2_context_dir which holds xml of the patients recored 

3. Download CliQuA annotations from Harvard DBMI under *Community Annotations Downloads*  
  

* Install requirements as follows:
```
pip install -r requirements.txt
```

## Experiments

### Preprocess data

- Generate evaluation data for CliQuA evaluation pipeline
1. *context_dir*: A path to a directory holding all original Golden Data Set XML files to be used for the evaluation.
2. *ground_truth_file*: A path to a file holding the CliQuA annotations. A path to a file holding the CliQuA annotations.
3. *output_file*: the preprocessed output file path.
   
Example usage:
``` 
    $ python preprocess_input.py  -context_dir "test_data/n2c2_context_dir" \
    -ground_truth_file "test_data/ground_truth_cliqua.csv" \§
    -output_file "test_data/inference_input_file.json"
```

 ### Evluation procedure

To evaluate LLM over the CliQua benchmark use the evaluation_flow as follow,
the script will save the annotated samples to the choosen path, The performance metrics above are printed to stdout in a comma separated text format.

1. *input_annotated_samples_file*: the output_file of the *preprocess_input.py* script.
2. *output_annotated_samples_file*: output path for the annotated samples.

Example usage:
```
    $ python evaluation_flow.py  -input_annotated_samples_file "test_data/inference_input_file.json" \
    -output_annotated_samples_file "test_data/inference_output_file.json"
```
The current evaluation framework apply *gemma-2b* uding the Hugging Face 
To apply you chosen LLM over the CliQua benchmark change LLM_inference method see template in `inference_utils.py`

