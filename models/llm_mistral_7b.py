from language_models import LanguageGeneration
from transformers import MistralForCausalLM, AutoTokenizer
from peft import PeftModel


class llm_mistral_small(LanguageGeneration):
    def __init__(self, path, pretrained=True, pre_version = "mistralai/Mistral-7B-Instruct-v0.3"):
        self.tokenizer = AutoTokenizer.from_pretrained(pre_version)
        self.model = MistralForCausalLM.from_pretrained(pre_version)
        if pretrained=False:
            self.model = PeftModel(self.model, path)
            self.model.merge_and_unload()
            
    
    def _generate_prompt(self, question, context=None):
        """Give the inputs, format them such that the input is compatible with instruct tuning"""
        pass
    
    def get_answer(self, question, context = None, max_length=128):
        refined_input = self._generate_prompt(question, context)
        input_ids = self.tokenizer(refined_input, return_tensors='pt')
        length = len(input_ids.input_ids)

        output_ids = model(input_ids.input_ids, max_length=max_length)
        generated_output = self.tokenizer.batch_decode(output_ids[length:], skip_special_tokens=True, clean_up+tokenizaton_spaceds=False)[0]
        return generated_output

        


