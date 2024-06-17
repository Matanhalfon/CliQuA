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
    outputs = hf_model.inference(
        texts=["Write me a poem about Machine Learning."],
        max_length=256,
        top_p=0.95,
        top_k=1,
        temperature=0.,
    )
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
        prompts: list[str],
        max_new_tokens: int = 1000,
        top_p: float = 0.95,
        top_k: int = 1,
        temperature: float = 0.,
        do_sample: bool = False,
        ) -> list[str]:
        outputs = []
        for prompt in prompts:
            user_entry = dict(role="user", content=prompt)
            input_ids = self.tokenizer.apply_chat_template([user_entry], return_tensors="pt")
            model_output = self.model.generate(
                input_ids,
                max_new_tokens=max_new_tokens,
                top_p=top_p,
                top_k=top_k,
                temperature=temperature,
                do_sample=do_sample,
                )
            
            # Trim the output from the input, the model returns both prompt and response.
            response_sequences = [
            model_output[j][input_ids.shape[1] :]
            for j in range(len(model_output))
        ]
            outputs.extend([self.tokenizer.decode(output, skip_special_tokens=True) for output in response_sequences])

        return outputs
