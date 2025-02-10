# -*- coding: utf-8 -*-

from src.utils import convert_2d_list_to_string
from src.models import ARCPair
from typing import List

def _load_prompt(prompt_name: str, language: str = "en") -> str:
    """
    Load a prompt from the prompts directory. If language is not 'en', use a language-specific prompt file.
    """
    if language != "en":
        prompt_name = f"{prompt_name}_{language}"
    return open(f"src/prompts/{prompt_name}.txt", "r").read()

def convert_task_pairs_to_prompt(training_pairs: List[ARCPair], test_input: ARCPair, language: str = "en") -> str:
    """
    Convert the training pairs to a prompt using the appropriate language prompt template
    """
    prompt_template = _load_prompt("system_prompt", language)

    # Define a dictionary for language-specific translations
    translations = {
        "en": {"example": "Example", "input": "INPUT", "output": "OUTPUT"},
        "cn": {"example": "\u793a\u4f8b", "input": "\u8f93\u5165", "output": "\u8f93\u51fa"},
        "jp": {"example": "\u4f8b", "input": "\u5165\u529b", "output": "\u51fa\u529b"}
    }

    trans = translations.get(language, translations["en"])
    example_text = trans["example"]
    input_text = trans["input"]
    output_text = trans["output"]

    training_examples = ""
    for i, pair in enumerate(training_pairs):
        training_examples += f"--{example_text} {i}-- \n\n {input_text}: \n\n"
        training_examples += convert_2d_list_to_string(pair.input) + "\n\n"
        training_examples += f"{output_text}: \n\n"
        training_examples += convert_2d_list_to_string(pair.output) + "\n\n"

    test_input = convert_2d_list_to_string(test_input.input)

    return prompt_template.format(training_examples=training_examples, test_input=test_input)