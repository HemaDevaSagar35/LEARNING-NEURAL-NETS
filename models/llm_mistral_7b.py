from language_models import LanguageGeneration
from transformers import MistralForCausalLM, AutoTokenizer
from peft import PeftModel
import torch


class llm_mistral_small(LanguageGeneration):
    def __init__(self, path, pretrained=True, pre_version = "mistralai/Mistral-7B-Instruct-v0.3"):
        self.tokenizer = AutoTokenizer.from_pretrained(pre_version)
        self.model = AutoModelForCausalLM.from_pretrained(pre_version, device_map = 'auto')
        if pretrained=False:
            self.model = PeftModel(self.model, path)
            self.model.merge_and_unload()


    def get_answer(self, refined_input, max_length=128):
        input_ids = self.tokenizer(refined_input, return_tensors='pt')
        input_ids.to(self.model.device)

        length = len(input_ids.input_ids)

        output_ids = model(input_ids.input_ids, max_length = max_length)
        generated_output = self.tokenizer.batch_decode(output_ids[length:], skip_special_tokens=True, clean_up_tokenizaton_spaces=False)[0]
        return generated_output

        


