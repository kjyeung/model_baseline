from .provider import ProviderAdapter
from openai import OpenAI
import json
import os
class OpenAICompatibleAdapter(ProviderAdapter):
    def __init__(self, model_name: str, base_url: str, max_tokens: int = 4024):
        self.client = self.init_client(base_url)
        self.model_name = model_name
        self.max_tokens = max_tokens

    def init_client(self, base_url: str):
        # No API key needed for local LLMs
        if not os.environ.get("OPENAI_API_KEY"):
            raise ValueError("OPENAI_API_KEY not found in environment variables")
        return OpenAI(base_url=base_url)

    def make_prediction(self, prompt: str) -> str:
        messages = [
            {"role": "user", "content": prompt}
        ]
        response = self.chat_completion(messages)
        return response.choices[0].message.content.strip()

    def chat_completion(self, messages: list) -> object:
        return self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            max_tokens=self.max_tokens,
        )

    def extract_json_from_response(self, input_response: str) -> list[list[int]] | None:
        prompt = f"""
You are a helpful assistant. Extract only the JSON of the test output from the following response. 
Do not include any explanation or additional text; only return valid JSON.

Response:
{input_response}

The JSON should be in this format:
{{
"response": [
    [1, 2, 3],
    [4, 5, 6]
]
}}
"""
        completion = self.client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
        )

        assistant_content = completion.choices[0].message.content.strip()

        # Some models like to wrap the response in a code block
        if assistant_content.startswith("```json"):
            assistant_content = "\n".join(assistant_content.split("\n")[1:])
        
        if assistant_content.endswith("```"):
            assistant_content = "\n".join(assistant_content.split("\n")[:-1])

        try:
            json_entities = json.loads(assistant_content)
            return json_entities.get("response")
        except json.JSONDecodeError:
            return None
