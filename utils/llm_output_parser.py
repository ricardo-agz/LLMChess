import os
import yaml
import json
import json5
from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel, ValidationError
from typing import (
    Type,
    TypeVar,
)


load_dotenv()

client = OpenAI(
    base_url="https://router.neutrinoapp.com/api/engines",
    api_key=os.getenv("NEUTRINO_API_KEY"),
)


def messages_to_str(messages: list[dict[str, str]]) -> str:
    prompt = ""
    for message in messages:
        prompt += f"{message['role']}:\n{message['content']}\n"

    return prompt


T = TypeVar("T", bound=BaseModel)


class OutputParserException(Exception):
    def __init__(self, message: str, llm_output: str) -> None:
        super().__init__(message)
        self.llm_output = llm_output


class PydanticOutputParser:
    """Parse an output using a pydantic model."""

    pydantic_object: Type[T]
    """The pydantic model to parse."""

    def __init__(self, pydantic_object: Type[T]) -> None:
        self.pydantic_object = pydantic_object

    def parse(self, text: str) -> T:
        json_str = self.extract_json(text)
        try:
            json_object = json5.loads(json_str)
            return self.pydantic_object.parse_obj(json_object)
        except json.JSONDecodeError:
            # If direct parsing fails, attempt to repair and parse
            repaired_text = self._repair_json(
                json_str
            )  # Apply repair on the extracted JSON
            try:
                json_object = json5.loads(repaired_text)
                return self.pydantic_object.parse_obj(json_object)
            except json.JSONDecodeError as e:
                name = self.pydantic_object.__name__
                msg = (
                    f"Failed to parse {name} from completion:\n{json_str}\n\nError: {e}"
                )
                raise OutputParserException(msg, llm_output=text)

    @staticmethod
    def extract_json(text: str) -> str:
        """This method extracts the first complete JSON object found in the text."""
        depth = 0
        start_index = -1
        for i, char in enumerate(text):
            if char == "{":
                if depth == 0:
                    start_index = i
                depth += 1
            elif char == "}":
                depth -= 1
                if depth == 0 and start_index != -1:
                    try:
                        json5.loads(text[start_index : i + 1])
                        return text[start_index : i + 1]
                    except json.JSONDecodeError:
                        continue
        raise ValueError("No valid JSON object found in the text.")

    @staticmethod
    def _repair_json(text: str) -> str:
        """Attempts to repair malformed JSON text to create a valid JSON."""
        try:
            lines = text.split("\n")
            repaired_lines = []
            for line in lines:
                # Remove common invalid characters and wrap keys/values in quotes
                cleaned_line = line.strip().replace("'", '"').rstrip(",")
                if ":" in cleaned_line:
                    parts = cleaned_line.split(":", 1)
                    key = json.dumps(parts[0].strip().strip('"'))
                    value = parts[1].strip().strip('"')
                    if value.isdigit():
                        repaired_lines.append(f"{key}: {value}")
                    else:
                        repaired_lines.append(f"{key}: {json.dumps(value)}")
                else:
                    repaired_lines.append(cleaned_line)
            repaired_json = "{\n" + ",\n".join(repaired_lines) + "\n}"
            json5.loads(repaired_json)
            return repaired_json
        except json.JSONDecodeError as e:
            raise ValueError(f"Failed to repair JSON: {str(e)}")

    def get_format_instructions(self) -> str:
        schema = self.pydantic_object.model_json_schema()

        # Remove extraneous fields.
        reduced_schema = schema
        if "title" in reduced_schema:
            del reduced_schema["title"]
        if "type" in reduced_schema:
            del reduced_schema["type"]
        # Ensure json in context is well-formed with double quotes.
        schema_str = json.dumps(reduced_schema)

        PYDANTIC_FORMAT_INSTRUCTIONS = """The output should be formatted as a JSON instance that conforms to the JSON schema below.

As an example, for the schema {{"properties": {{"foo": {{"title": "Foo", "description": "a list of strings", "type": "array", "items": {{"type": "string"}}}}}}, "required": ["foo"]}}
the object {{"foo": ["bar", "baz"]}} is a well-formatted instance of the schema. The object {{"properties": {{"foo": ["bar", "baz"]}}}} is not well-formatted.

Here is the output schema:
```
{schema}
```"""

        return PYDANTIC_FORMAT_INSTRUCTIONS.format(schema=schema_str)

    def get_example_str(self) -> str:
        return self.pydantic_object.get_example_str()

    @property
    def _type(self) -> str:
        return "pydantic"


class PydanticYAMLOutputParser:
    """Parse an output using a pydantic model."""

    pydantic_object: Type[T]
    """The pydantic model to parse."""

    def __init__(self, pydantic_object: Type[T]) -> None:
        self.pydantic_object = pydantic_object

    def parse(self, text: str) -> T:
        yaml_str = self.extract_yaml(text)
        try:
            yaml_object = yaml.safe_load(yaml_str)
            try:
                return self.pydantic_object.parse_obj(yaml_object)
            except ValidationError as e:
                name = self.pydantic_object.__name__
                msg = (
                    f"Failed to parse {name} from completion:\n{yaml_str}\n\nError: {e}"
                )
                raise OutputParserException(msg, llm_output=text)
        except yaml.YAMLError:
            # If direct parsing fails, attempt to repair and parse
            repaired_text = self._repair_yaml(
                yaml_str
            )  # Apply repair on the extracted YAML
            try:
                yaml_object = yaml.safe_load(repaired_text)
                return self.pydantic_object.parse_obj(yaml_object)
            except yaml.YAMLError as e:
                name = self.pydantic_object.__name__
                msg = (
                    f"Failed to parse {name} from completion:\n{yaml_str}\n\nError: {e}"
                )
                raise OutputParserException(msg, llm_output=text)

    @staticmethod
    def extract_yaml(text: str) -> str:
        str_lines = text.split("\n")
        start_index = -1
        end_index = -1
        for i, line in enumerate(str_lines):
            if start_index != -1 and line.strip() == "```":
                end_index = i
                break
            if line.strip() == "```yaml" or line.strip() == "```":
                start_index = i

        if (start_index == -1 or end_index == -1) and not (start_index == end_index):
            raise ValueError(
                f"No valid YAML object found in the text. Code block parsing failed.\n{text}"
            )
        elif start_index == end_index:
            start_index = 0
            end_index = len(str_lines)

        yaml_str = "\n".join(str_lines[start_index + 1 : end_index])
        try:
            yaml.safe_load(yaml_str)
            return yaml_str
        except yaml.YAMLError as e:
            raise ValueError(
                f"No valid YAML object found in the text.\n{yaml_str}\n{e}"
            )

    @staticmethod
    def _is_yaml_line(line: str):
        return "- " in line or ": " in line

    @staticmethod
    def _repair_yaml(text: str) -> str:
        """Attempts to repair malformed YAML text to create a valid YAML."""
        try:
            lines = text.split("\n")
            yaml_lines = [
                line for line in lines if PydanticYAMLOutputParser._is_yaml_line(line)
            ]
            yaml_str = "\n".join(yaml_lines)

            yaml.safe_load(yaml_str)  # Validate repaired YAML
            return yaml_str
        except yaml.YAMLError as e:
            raise ValueError(f"Failed to repair YAML: {str(e)}")

    def get_format_instructions(self) -> str:
        json_schema = self.pydantic_object.model_json_schema()

        # Remove extraneous fields.
        reduced_schema = json_schema
        if "title" in reduced_schema:
            del reduced_schema["title"]
        if "type" in reduced_schema:
            del reduced_schema["type"]
        # Ensure YAML in context is well-formed.
        schema_str = yaml.safe_dump(reduced_schema)

        PYDANTIC_FORMAT_INSTRUCTIONS = """The output should be formatted as a VALID YAML instance that conforms to the YAML schema below.

        As an example, for the schema:
        properties:
          foo:
            title: Foo
            description: a list of strings
            type: array
            items:
              type: string
        required:
          - foo

        the object:
        ```yaml
        foo:
          - bar
          - baz
        ```
        is a well-formatted instance of the schema. 
        The object:
        ```yaml
        properties:
          foo:
            - bar
            - baz
        ```
        is not well-formatted.

        Here is the output schema:
        ```yaml
        {schema}
        ```

        Your output MUST be wrapped in a code block with the language set to 'yaml'.
        For example:
        ```yaml
        foo:
            - bar
            - baz
        ```

        The content inside the code block MUST be a valid YAML instance that conforms to the schema and should be able to be parsed by a YAML parser.
        """

        return PYDANTIC_FORMAT_INSTRUCTIONS.format(schema=schema_str)

    def get_example_str(self) -> str:
        return f"```yaml\n{self.pydantic_object.get_example_str(to_yaml=True)}\n```"

    @property
    def _type(self) -> str:
        return "pydantic"


class OutputFixingParser:
    """
    A parser that attempts to fix outputs based on the constraints specified by another parser.
    If parsing fails, it retries using the language model to correct the output.
    """

    NAIVE_FIX_WITH_INSTRUCTIONS = """
You are an expert AI trained to reformat and fix responses that failed the satisfy specific formatting requirements
and were unable to be parsed. You will be given the original formatting instructions, the wrongly-formatted
response, and the error message. Your task is to reformat the response so that it satisfies the given instructions
and is able to be parsed. 

These were the original instructions:
--------------
{instructions}
--------------
Formatting Requirements:
--------------
{formatting_requirements}
--------------
Completion:
--------------
{completion}
--------------

Above, the Completion did not satisfy the constraints given in the Instructions.
Error:
--------------
{error}
--------------

Respond with a corrected completion that satisfies the original instructions and formatting requirements."""

    NAIVE_FIX = """
You are an expert AI trained to reformat and fix responses that failed the satisfy specific formatting requirements
and were unable to be parsed. You will be given a wrongly-formatted response along with an error message. 
Your task is to reformat the response so that it satisfies the formatting requirements and is able to be parsed. 

Formatting Requirements:
--------------
{formatting_requirements}
--------------
Completion:
--------------
{completion}
--------------

Above, the Completion did not satisfy the constraints given in the Instructions.
Error:
--------------
{error}
--------------

Respond with a corrected completion that satisfies the original instructions and formatting requirements."""

    def __init__(
        self,
        parser: PydanticOutputParser | PydanticYAMLOutputParser,
        model: str = "gpt-3.5-turbo",
        max_retries: int = None,
        last_try_model: str = None,
        retry_models: list[str] = (
            "neutrino-cohere-command-r",
            "anthropic.claude-3-haiku-20240307-v1:0",
            "microsoft/WizardLM-2-8x22B",
            "neutrino-cohere-command-r-plus",
        ),
    ) -> None:
        """
        Initializes the OutputFixingParser.

        :param parser: An instance of BaseOutputParser to parse the initial response.
        :param model: The language model to be used for fixing the response.
        :param max_retries: The maximum number of retries to attempt fixing the response.
        :param last_try_model: The language model to use for the last attempt.
        """
        self._parser = parser
        self._models_to_use = [model] + list(retry_models)
        self._retry_models = retry_models
        self._max_retries = max_retries or len(retry_models) + 1
        self._last_try_model = last_try_model or retry_models[-1]

    def get_format_instructions(self) -> str:
        return self._parser.get_format_instructions()

    def get_example_str(self) -> str:
        return self._parser.get_example_str()

    def parse(
        self,
        response_text: str,
        original_messages: list[dict[str, str]] = None,
        with_metadata: bool = False,
    ) -> T | tuple[T, dict[str, any]]:
        """
        Parses the given response text, retrying with language model assistance if parsing fails.

        :param response_text: The text response to parse.
        :param original_messages: The original messages that led to the response.
        :param with_metadata: A flag indicating whether to return metadata.
        :return: The parsed evaluation if successful.
        :raises: The last exception if all retries fail.
        """
        attempt_count = 0
        current_response = response_text
        original_error = None
        running_cost = 0.0
        running_latency = 0.0

        if original_messages is not None:
            # try first without original instructions then with original instructions
            models_to_use = [
                self._models_to_use[0]
            ] + self._models_to_use  # try the first model twice
            max_retries = self._max_retries + 1
        else:
            models_to_use = self._models_to_use
            max_retries = self._max_retries

        while True:
            try:
                if with_metadata:
                    return (
                        self._parser.parse(current_response),
                        {
                            "cost": running_cost,
                            "latency": running_latency,
                        },
                    )
                else:
                    return self._parser.parse(current_response)

            except Exception as e:
                if original_error is None:
                    original_error = str(e)

                if attempt_count >= max_retries:
                    raise e  # Re-raise the last exception

                model_to_use = models_to_use[attempt_count]

                attempt_count += 1

                if attempt_count == 0:
                    print(f"Attempt {attempt_count} failed: {e}")
                else:
                    print(
                        f"Attempt {attempt_count} failed using retry model: {model_to_use}: {e}"
                    )
                print("Incorrectly formatted response:")
                print("-" * 25)
                print(current_response)
                print("=" * 25)

                # Switch to last try model for the last attempt
                if attempt_count == max_retries - 1:
                    print(
                        f"Switching to {self._last_try_model} for the last attempt..."
                    )
                    model_to_use = self._last_try_model

                input_messages = self._construct_input_messages(
                    response_text=response_text,
                    original_messages=original_messages,
                    error_message=original_error,
                    attempt_count=attempt_count,
                )

                answer = client.chat.completions.create(
                    model=model_to_use,
                    messages=input_messages,
                )
                current_response = answer.choices[0].message.content

                cost = answer.metadata["cost"]
                latency = answer.metadata["latency"]
                running_cost += cost
                running_latency += latency

                print("Retrying with corrected response...")

    def _construct_input_messages(
        self,
        response_text: str,
        error_message: str,
        original_messages: list[dict[str, str]] = None,
        attempt_count: int = None,
    ) -> list:
        """
        Constructs input messages for the language model based on the template, error, and instructions.

        :param response_text: The initial response text that failed parsing.
        :param error_message: The error message obtained from the failed parsing.
        :return: A list containing the input messages for the language model.
        """
        if original_messages is None or attempt_count == 0:
            return [
                {
                    "role": "user",
                    "content": self.NAIVE_FIX.replace(
                        "{formatting_requirements}",
                        self._parser.get_format_instructions(),
                    )
                    .replace("{completion}", response_text)
                    .replace("{error}", error_message),
                },
            ]

        original_instructions = messages_to_str(original_messages)
        return [
            {
                "role": "user",
                "content": self.NAIVE_FIX_WITH_INSTRUCTIONS.replace(
                    "{formatting_requirements}", self._parser.get_format_instructions()
                )
                .replace("{completion}", response_text)
                .replace("{error}", error_message)
                .replace("{instructions}", original_instructions),
            },
        ]
