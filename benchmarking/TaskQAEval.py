from .utils.metrics import * 
from .utils.prompting import *

class EvaluateOnQA(EvaluateOnTask):

    # def get_answer(input_text):
    #     inputs = tokenizer(input_text, return_tensors="pt", truncation=True)
    #     with torch.no_grad():
    #         outputs = model(**inputs)
    #     start_logits, end_logits = outputs.start_logits, outputs.end_logits
    #     start_index = torch.argmax(start_logits)
    #     end_index = torch.argmax(end_logits) + 1
    #     return tokenizer.decode(inputs['input_ids'][0][start_index:end_index])
    #I don't think we should be writing the get_answer here. The model abstraction should have the get_answer to it else it would be hard

    def evaluate(self, task, fetch_fn, type = "validation"):
        #fetch_fn is a function to fetch appropriate key values from the dataset object
        dataset = load_dataset(task)

        predictions = []
        references = []

        for example in dataset["validation"]:
            question, context, answer_list = fetch_fn(example)
            prompt = mistral_prompt(question, context, task = "QA")
            predicted_answer = model.get_answer(prompt)

            predictions.append(predicted_answer)
            references.append(answer_list)
        
        stats = compute_metrics_multi(predictions, references)
        #print(stats)
        return stats
