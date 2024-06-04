"""This module contains classes for API interaction with language model inference."""
from typing import Protocol
from transformers import AutoTokenizer, AutoModelForCausalLM

class LLMInference(Protocol):

    def inference(self, texts: list[str]) -> list[str]:
        """Inference method for language models.
        
        Args:
            texts: List of input texts.
        
        Returns:
            List of generated texts.
        """


class HFInference:
    """Hugging Face model inference class for language models.
    
      ```python
    model_name = "google/gemma-2b"
    hf_model = HFInference(model_name)
    outputs = hf_model.inference(["Write me a poem about Machine Learning."])
    ```
    """
    def __init__(self, model_name: str):
        """
        
        Args:
            model_name: Hugging Face model name.
        """
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)

    def inference(
        self,
        texts: list[str],
        max_length: int = 500,
        batch_size: int = 16,
        ) -> list[str]:
        batches = [texts[i:i + batch_size] for i in range(0, len(texts), batch_size)]
        outputs = []
        for batch in batches:
            input_ids = self.tokenizer(batch, return_tensors="pt")
            model_output = self.model.generate(**input_ids, max_length=max_length)
            outputs.extend([self.tokenizer.decode(output, skip_special_tokens=True) for output in model_output])

        return outputs