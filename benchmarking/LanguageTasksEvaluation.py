from transformers import AutoTokenizer
from abc import ABC, abstractmethod
from typing import Any, String
from .models import LanguageGeneration

class EvaluateOnTask(ABC):
    model : LanguageGeneration

    def __init__(self, model: LanguageGeneration):
        self.model = model

    @abstractmethod
    def evaluate(self, task):
        raise NotImplementedError("Subclasses should implement this method")
    
