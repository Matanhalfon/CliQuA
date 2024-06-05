# CliQuA: A Benchmark for Clinical Records Question Answering
An eval framework that enables LLM inference over the new dataset release of the paper - "CliQuA: A Benchmark for Clinical Records Question Answering"


## Prerequisites
* Download N2C2  [data](url) and CliQuA annotations [file](url)
  

* Install requirements as follows:
```
pip install -r requirements.txt
```

## Experiments
Run  


* Generate evaluation data for CliQuA evaluation pipeline.
Example usage:
``` 
    $ python preprocess_input.py  -context_dir "test_data/n2c2_context_dir" \
    -ground_truth_file "test_data/ground_truth_cliqua.csv" \§
    -output_file "test_data/inference_input_file.json"
```

* To change LLM_inference see template in `inference_utils.py`
*  Run evaluation flow, which creates prompts based on the preprocessed
data above and logs parsed inference output in addition to running the CliQuA
evaluation. Alter `run_llm` arguments to use another llm.
Example usage:
```
    $ python evaluation_flow.py  -input_annotated_samples_file "test_data/inference_input_file.json" \
    -output_annotated_samples_file "test_data/inference_output_file.json"
```