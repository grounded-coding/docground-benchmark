import os
import openai
import re
from lleval.scorer import PromptScorer
from lleval.evaluator import PromptTemplate

class OpenAIScorer(PromptScorer):
    def __init__(self, metric_config_file, prompt_template: PromptTemplate, num_retries=3):
        super().__init__("", metric_config_file, prompt_template, num_retries)

    def build_and_submit_prompt(self, i, output_list, src_list, context_list, dimension, method="likert"):
        prompt = self.prompt_template.get_prompt(dimension, output_list[i], src_list[i], context_list[i])
        openai.api_key = os.environ["OPENAI_API_KEY"]

        temperature = self.metric_config["gen_params"]["temperature"]

        success = False
        for _ in range(self.num_retries):
            success = False
            response_text = openai.ChatCompletion.create(model="gpt-3.5-turbo",
                                                         temperature=temperature,
                                                         messages=[{"role": "user", "content": prompt}])
            response_text = response_text["choices"][0]["message"]["content"]

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