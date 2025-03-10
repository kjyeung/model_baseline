import json
from src.adapters import ProviderAdapter, AnthropicAdapter, OpenAIAdapter, DeepseekCoTAdapter
from dotenv import load_dotenv
import src.utils as utils
from src.models import ARCTaskOutput, ARCPair
from src.prompts.prompt_manager import convert_task_pairs_to_prompt
from typing import List, Any, Optional
import os
import argparse

load_dotenv()

class ARCTester:
    def __init__(self, provider: str, generation_model_name: str, extraction_model_name: str, generation_base_url: str, extraction_base_url: str, parse_cot: bool, save_submission_dir: str, overwrite_submission: bool, print_submission: bool, num_attempts: int, retry_attempts: int, print_logs: bool, language: str):
        self.provider = self.init_provider(provider, generation_model_name, extraction_model_name, generation_base_url, extraction_base_url, parse_cot)
        self.save_submission_dir = save_submission_dir
        self.overwrite_submission = overwrite_submission
        self.print_submission = print_submission
        self.num_attempts = num_attempts
        self.retry_attempts = retry_attempts
        self.print_logs = print_logs
        self.language = language

    def init_provider(self, provider: str, generation_model_name: str, extraction_model_name: str, generation_base_url: str, extraction_base_url: str, parse_cot: bool) -> ProviderAdapter:
        if provider == "anthropic":
            return AnthropicAdapter(generation_model_name)
        elif provider == "openai":
            return OpenAIAdapter(generation_model_name)
        elif provider == "deepseek":
            return DeepseekCoTAdapter(generation_model_name, generation_base_url, extraction_model_name, extraction_base_url, parse_cot)
        else:
            raise ValueError(f"Unsupported provider: {provider}")
        
    def print_log(self, message: str):
        if self.print_logs:
            print(message)

    def convert_single_integer_to_2d_list(self, data: str) -> Optional[List[List[int]]]:
        """
        If the input string represents a single integer, return it as a nested list.
        Otherwise, return None.
        """
        try:
            parsed_data = int(data)
            result = [[parsed_data]]
            return result
        except ValueError:
            pass
        return None

    def convert_1d_list_to_2d_list(self, data: str) -> Optional[List[List[int]]]:
        """
        If the input string represents a single-item list containing one or more integers,
        return it as a nested list. Otherwise, return None.
        """
        try:
            # Remove whitespace and parse the string as JSON
            parsed_data = json.loads(data.strip())
            if isinstance(parsed_data, list) and 1 <= len(parsed_data) <= 30 and all(isinstance(item, int) for item in parsed_data):
                result = [[item] for item in parsed_data]
                return result
        except json.JSONDecodeError:
            pass
        return None

    def extract_json_from_response(self, response: str) -> List[List[int]]:
        """
        Extract JSON from various possible formats in the response.
        """
        # Try to extract JSON array using regex
        json_str_match = utils.regex_extract_json(response)
        if json_str_match:
            return json_str_match
        
        # Try to extract JSON from code block
        json_code_block_match = utils.extract_json_from_code_block(response)
        if json_code_block_match:
            return json_code_block_match

        # Finally, use an LLM to extract the JSON
        json_llm_match = self.provider.extract_json_from_response(response)
        if json_llm_match:
            return json_llm_match
    
        # If all extraction methods fail, raise an exception
        raise json.JSONDecodeError("Failed to extract valid JSON from the response", response, 0)

    def parse_and_validate_json(self, response: str) -> ARCTaskOutput:
        """
        Parse the response string into JSON and validate its structure.

        This is unfortunately a necessary evil.
        """
        single_integer_match = self.convert_single_integer_to_2d_list(response)
        if single_integer_match:
            return single_integer_match

        # Try to convert 1d list to 2d list
        # This is a band-aid hack when the LLM returns a single-item list containing an integer
        one_d_match = self.convert_1d_list_to_2d_list(response)
        if one_d_match:
            return one_d_match

        # First, try to parse the raw JSON response
        try:
            parsed_json = json.loads(response)
        except:
            # If raw parsing fails, try to extract JSON from various formats
            parsed_json = self.extract_json_from_response(response)
        
        # Validate the structure of the parsed JSON
        if not isinstance(parsed_json, list) or not all(isinstance(row, list) for row in parsed_json):
            raise ValueError("Invalid JSON structure: expected a list of lists")
        
        return parsed_json
        
    def predict_task_output(self, training_pairs: List[ARCPair], test_input: ARCPair):
        """
        Given a task, predict the test output. This reponse may need parsing.

        Convert the training pairs and test pairs into a prompt
        Give the prompt to the LLM
        return the response
        """

        # Convert the training pairs and test pairs into a prompt
        prompt = convert_task_pairs_to_prompt(training_pairs, test_input, language=self.language)

        self.print_log(f"Making prediction for task")
        answer, chain_of_thought = self.provider.make_prediction(prompt)

        self.print_log(f"Answer: {answer}")

        return answer, chain_of_thought

    def get_task_prediction(self, training_pairs: List[ARCPair], test_input: ARCPair) -> ARCTaskOutput:
        """
        challenge_tasks: dict a list of tasks
        task_id: str the id of the task we want to get a prediction for
        test_input_index: the index of your test input. 96% of tests only have 1 input.

        Given a task, predict the test output
        """

        # Get the string representation of your task
        answer, chain_of_thought = self.predict_task_output(training_pairs, test_input)

        # Attempt to parse and validate the JSON response
        self.print_log("Parsing JSON response.")
        try:
            # Attempt to parse and validate the JSON response
            json_response = self.parse_and_validate_json(answer)
        except json.JSONDecodeError as e:
            self.print_log(f"JSON parsing failed: {e}")
            self.print_log(f"LLM response received but JSON parsing failed. Response: '{answer}'")
            # Handle the error (e.g., return a default value, retry, or raise an exception)
            raise
        
        return json_response, chain_of_thought, answer
    
    def generate_task_solution(self, data_dir, task_id):
        """
        data_dir: str, the directory of the data set to run
        task_id: str, the specific task to run. If None, run all tasks.
        num_attempts: int the number of times to attempt a prediction. The official competition has 2 attempts.
        retry_attempts: int the number of times to retry a prediction if it fails
        save_submission: bool, whether to save the submission to a file after each task
        """
        
        self.print_log(f"Running task {task_id}")
        utils.validate_data(data_dir, task_id)

        # Logic for overwrite. If save_submission_dir is provided, check if the submission already exists
        if self.save_submission_dir and utils.submission_exists(self.save_submission_dir, task_id) and not self.overwrite_submission:
            self.print_log(f"Submission for task {task_id} already exists, skipping")
            return
        
        task_attempts = []

        train_pairs = utils.get_train_pairs_from_task(data_dir, task_id)
        test_input = utils.get_test_input_from_task(data_dir, task_id)

        # Go through each test pair to get a prediction. 96% of challenges have 1 pair.
        for t, pair in enumerate(test_input):
            self.print_log(f"Starting task {task_id}, Pair #{t+1}")

            # Dictionary to store attempts for the current test pair
            pair_attempts = {}

            # Run through each prediction attempt
            for attempt in range(1, self.num_attempts + 1):
                attempt_key = f"attempt_{attempt}"
                cot_key = f"cot_{attempt}"
                initial_response_key = f"initial_response_{attempt}"
                pair_attempts[attempt_key] = None  # Initialize as None
                pair_attempts[cot_key] = None
                pair_attempts[initial_response_key] = None

                # Try to get a prediction, with retries in case of failure
                for retry in range(self.retry_attempts):
                    try:
                        self.print_log(f"    Predicting attempt #{attempt}, retry #{retry + 1}")
                        prediction, chain_of_thought, initial_response = self.get_task_prediction(
                            training_pairs=train_pairs,
                            test_input=pair
                        )

                        if prediction is not None:
                            self.print_log(f"    Prediction: {prediction}")
                            # self.print_log(f"    Chain of thought: {chain_of_thought}")
                            pair_attempts[attempt_key] = prediction
                            pair_attempts[cot_key] = chain_of_thought
                            pair_attempts[initial_response_key] = initial_response
                            break  # Break the retry loop if prediction is successful
                        else:
                            self.print_log("    Prediction returned None, possibly due to rate limiting")
                    except Exception as e:
                        self.print_log(f"Retrying: {e}")

                    if retry == self.retry_attempts - 1:
                        self.print_log(f"    All retries failed for attempt #{attempt}")

            # Only append non-None attempts
            if any(v is not None for v in pair_attempts.values()):
                task_attempts.append(pair_attempts)

        if task_attempts:
            if self.print_submission:
                self.print_log(f"Submission for task {task_id}:\n{task_attempts}")

            if self.save_submission_dir:
                utils.save_submission(self.save_submission_dir, task_id, task_attempts)
        else:
            self.print_log(f"No valid predictions for task {task_id}, skipping submission")

        return task_attempts if task_attempts else None
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run ARC Tester")
    parser.add_argument("--data_dir", type=str, help="Data set to run. Configure in config/config.json")
    parser.add_argument("--task_id", type=str, help="Specific task ID to run")
    parser.add_argument("--provider", type=str, default="anthropic", help="Provider to use")
    parser.add_argument("--generation_model", type=str, default="claude-3-5-sonnet-20241022", help="Model to use for generation")
    parser.add_argument("--extraction_model", type=str, default=None, help="Model to use for extraction")
    parser.add_argument(
        "--save_submission_dir",
        type=str,
        metavar="FOLDER_NAME",
        help="Folder name to save the submissions under Ex: 'submissions/claude-3-5-sonnet-20241022'"
    )
    parser.add_argument("--overwrite_submission", action="store_true", help="Overwrite the submission if it already exists")
    parser.add_argument("--print_submission", action="store_true", help="Print the submission to the console after each task")
    parser.add_argument("--task_set", type=str, default="public_eval", choices=["public_eval", "public_training"], help="Task set to run")
    parser.add_argument("--num_attempts", type=int, default=2, help="Number of attempts for each prediction")
    parser.add_argument("--retry_attempts", type=int, default=2, help="Number of retry attempts for failed predictions")
    parser.add_argument("--print_logs", action="store_true", help="Disable printing logs to console (default: False)")
    parser.add_argument("--generation_base_url", type=str, default="https://api.deepseek.com", help="Deepseek CoT: Base URL used for generation")
    parser.add_argument("--extraction_base_url", type=str, help="Deepseek CoT: Base URL used for extraction")
    parser.add_argument("--parse_cot", action="store_true",
                        help="Deepseek CoT: Whether to extract chain-of-thought without API support")
    parser.add_argument("--language", type=str, choices=["en", "cn", "jp"], default="en", help="Language for the prompt")
    
    args = parser.parse_args()

    arc_solver = ARCTester(
        provider=args.provider,
        generation_model_name=args.generation_model,
        extraction_model_name=args.extraction_model,
        generation_base_url=args.generation_base_url,
        extraction_base_url=args.extraction_base_url,
        parse_cot=args.parse_cot,
        save_submission_dir=args.save_submission_dir, 
        overwrite_submission=args.overwrite_submission,
        print_submission=args.print_submission,
        num_attempts=args.num_attempts,
        retry_attempts=args.retry_attempts,
        print_logs=args.print_logs,
        language=args.language
    )
   
    arc_solver.generate_task_solution(
        data_dir=args.data_dir,
        task_id=args.task_id, 
    )