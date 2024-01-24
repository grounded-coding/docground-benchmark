import os
from openai import OpenAI
import re
from lleval.scorer import PromptScorer
from lleval.evaluator import PromptTemplate

class OpenAIScorer(PromptScorer):
    def __init__(self, metric_config_file, prompt_template: PromptTemplate, gpt_model="gpt-4-1106-preview", num_retries=3):
        super().__init__("", metric_config_file, prompt_template, num_retries)
        self.gpt_model = gpt_model

    def build_and_submit_prompt(self, i, output_list, src_list, context_list, dimension, method="likert"):
        prompt = self.prompt_template.get_prompt(dimension, output_list[i], src_list[i], context_list[i])
        client = OpenAI()

        temperature = self.metric_config["gen_params"]["temperature"]

        # Return the length of the prompt in characters divided by 4
        print("Prompt length:", len(prompt) // 4)

        success = False
        for _ in range(self.num_retries):
            success = False
            response_text = client.chat.completions.create(model=self.gpt_model,
                                                         temperature=temperature,
                                                         messages=[{"role": "user", "content": prompt}])
            response_text = response_text.choices[0].message.content

            if method == "winrate":
                regex_str = rf'\nMore {dimension["name"]} response: ([1-2])'
            else:
                regex_str = rf'\n{dimension["name"].capitalize()} Score: ([12345])'

            match = re.search(regex_str, response_text)
            if match:
                winner = match.group(1)
                explanation = response_text[:match.start()].rstrip("\n").lstrip("\n")
                success = True
                break
        if not success:
            winner, explanation = -1, "Error in syntax retrieval"

        return {dimension["name"]: float(winner), "id": i, "explanation": explanation}