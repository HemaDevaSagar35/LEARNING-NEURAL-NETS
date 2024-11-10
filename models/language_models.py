from typing import Any
from abc import ABC, abstractmethod
from transformers import AutoTokenizer

class LanguageGeneration(ABC):
    # model : Any
    # tokenizer : AutoTokenizer

    # def __init__(self, tokenizer: AutoTokenizer, model: Any):
    #     self.tokenizer = tokenizer
    #     self.model = model

    #     # Runtime check to ensure the model has the expected methods
    #     if not (hasattr(model, '__call__') or hasattr(model, 'forward')):
    #         raise ValueError("The model should have a __call__ or forward method for inference.")
    

    @abstractmethod
    def get_answer(self):
        raise NotImplementedError("Subclasses should implement this method")