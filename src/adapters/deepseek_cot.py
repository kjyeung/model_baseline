from .provider import ProviderAdapter
import os
from dotenv import load_dotenv
import json
from openai import OpenAI
load_dotenv()
import re

class DeepseekCoTAdapter(ProviderAdapter):
    def __init__(self, generation_model_name: str, generation_base_url: str , extraction_model_name: str = None, extraction_base_url: str = None, parse_cot:bool = False, max_tokens: int = 4024):
        """
        Initialize the flexible Deepseek adapter.
        
        Args:
            generation_model_name (str): Name of the model.
            generation_base_url (str): Base URL used for generating responses.
            extraction_model_name (Optional[str]): Name of the model.
                Defaults to generation_model_name if not provided.
            extraction_base_url (Optional[str]): Base URL used for JSON extraction.
                Defaults to generation_base_url if not provided.
            parse_cot (bool): Whether or not to parse a chain-of-thought from the response.
            max_tokens (int): Maximum tokens for generation.
        """
        self.generation_model_name = generation_model_name
        self.extraction_model_name = extraction_model_name if extraction_model_name else generation_model_name
        self.max_tokens = max_tokens
        self.parse_cot = parse_cot
        self.generation_base_url = generation_base_url
        # If a separate extraction URL is not specified, fall back to using the generation URL
        self.extraction_base_url = extraction_base_url if extraction_base_url else generation_base_url

        self.gen_client = self.init_client(self.generation_base_url)
        # Only create a separate client if the extraction base URL is different
        if self.extraction_base_url != self.generation_base_url:
            self.extraction_client = self.init_client(self.extraction_base_url)
        else:
            self.extraction_client = self.gen_client

    def init_client(self, url: str):
        if not os.environ.get("DEEPSEEK_API_KEY"):
            raise ValueError("DEEPSEEK_API_KEY not found in environment variables")
        
        client = OpenAI(api_key = os.environ.get("DEEPSEEK_API_KEY"), base_url=url)
        return client

    def make_prediction(self, prompt: str) -> str:
        messages = [
            {"role": "user", "content": prompt}
        ]
        # print(f"Prompt: {prompt}")
        response = self.chat_completion(messages)
        # print(response)
        # Uncomment to print token usage
        # print(f"USAGE|PROMPT|{response.usage.prompt_tokens}")
        # print(f"USAGE|COMPLETION|{response.usage.completion_tokens}")
        # print(f"Response: {response.choices[0].message.content.strip()}")
        print(f"Response: {response.choices[0].message}")
        print(f"Raw content: '{response.choices[0].message.content}'")
        if self.parse_cot:
            chain_of_thought, answer = self.extract_chain_of_thought(response)
        else:
            chain_of_thought = response.choices[0].message.reasoning_content
            answer = response.choices[0].message.content.strip()
        print(f"Extracted answer: '{answer}'")
        return answer, chain_of_thought

    def chat_completion(self, messages: str) -> str:
        # print(messages)
        # TODO: set temperature to 0.6 if using local model
        return self.gen_client.chat.completions.create(
            model=self.generation_model_name,
            temperature=0.6,
            # TODO: parameterize the reasoning_effort (including not setting it since it's only supported
            # o1, as of 12/19/2024)
            # Default value for o1 is 'medium'.
            # Uncomment to set a different value.
            # reasoning_effort='high',
            messages=messages,
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

        # print(f"Input response: {input_response}")
        completion = self.extraction_client.chat.completions.create(
            model=self.extraction_model_name,
            messages=[{"role": "user", "content": prompt}],
        )
        # Uncomment to print token usage 
        # print(f"USAGE|PROMPT|{completion.usage.prompt_tokens}")
        # print(f"USAGE|COMPLETION|{completion.usage.completion_tokens}")

        assistant_content = completion.choices[0].message.content.strip()

        # Some oai models like to wrap the response in a code block
        if assistant_content.startswith("```json"):
            assistant_content = "\n".join(assistant_content.split("\n")[1:])
        
        if assistant_content.endswith("```"):
            assistant_content = "\n".join(assistant_content.split("\n")[:-1])

        # Attempt to parse the returned content as JSON
        # print(f"For input response: {input_response}, got extracted content: {assistant_content}")
        try:
            json_entities = json.loads(assistant_content)
            return json_entities.get("response")
        except json.JSONDecodeError:
            return None

    def extract_chain_of_thought(self, response: str) -> str:
            """
            Separate content within <think>...</think> tags from the rest of the text.

            Args:
                text (str): The input string containing <think> blocks.

            Returns:
                tuple:
                    - think_contents: A list of strings extracted from within each <think> block.
                    - other_text: The remaining text with all <think> blocks removed.
            """
            # Find all non-overlapping occurrences inside <think>...</think>
            chain_of_thought =  [s.lstrip('\n') for s in re.findall(r'<think>(.*?)</think>', response, flags=re.DOTALL)]
            # Remove the <think> blocks from the text
            answer = re.sub(r'<think>.*?</think>', '', response, flags=re.DOTALL).lstrip('\n')
            
            return chain_of_thought, answer