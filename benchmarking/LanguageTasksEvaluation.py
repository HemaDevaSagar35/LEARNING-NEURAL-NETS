from transformers import AutoTokenizer
from abc import ABC, abstractmethod
from typing import Any, String

class EvaluateOnTask(ABC):
    tokenizer : AutoTokenizer
    model : Any

    def __init__(self, tokenizer: AutoTokenizer, model: Any):
        self.tokenizer = tokenizer
        self.model = model
        self.task = task

        # Runtime check to ensure the model has the expected methods
        if not (hasattr(model, '__call__') or hasattr(model, 'forward')):
            raise ValueError("The model should have a __call__ or forward method for inference.")

    @abstractmethod
    def evaluate(self, task):
        raise NotImplementedError("Subclasses should implement this method")
    
