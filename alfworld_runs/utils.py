import os
import sys # Used to check the Python version.
import openai
from tenacity import ( # A library for adding retry behavior to functions.
    retry, # A decorator to retry a function if it fails.
    stop_after_attempt, # type: ignore
    wait_random_exponential, # type: ignore
)

from typing import Optional, List # typing: Used for type hints. Optional: Indicates a variable can be of a specified type or None.
if sys.version_info >= (3, 8): # If Python ≥ 3.8, imports Literal from typing.
    from typing import Literal
else:
    from typing_extensions import Literal # Literal: Used to specify exact string values a variable can take.

# Defines a type Model that can only be one of the three specified strings (GPT-4, GPT-3.5 Turbo, or text-davinci-003).
Model = Literal["gpt-4", "gpt-3.5-turbo", "text-davinci-003"]

openai.api_key = os.getenv('OPENAI_API_KEY')

# Waits between 1 and 60 seconds (exponentially increasing). Stops after 6 attempts.
@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
def get_completion(prompt: str, temperature: float = 0.0, max_tokens: int = 256, stop_strs: Optional[List[str]] = None) -> str: # For older completion models (text-davinci-003).
    response = openai.Completion.create(
        model='text-davinci-003',
        prompt=prompt,
        temperature=temperature,
        max_tokens=max_tokens,
        top_p=1,
        frequency_penalty=0.0,
        presence_penalty=0.0,
        stop=stop_strs,
    )
    return response.choices[0].text

# For chat models (gpt-4, gpt-3.5-turbo).
@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
def get_chat(prompt: str, model: Model, temperature: float = 0.0, max_tokens: int = 256, stop_strs: Optional[List[str]] = None, is_batched: bool = False) -> str:
    assert model != "text-davinci-003"
    messages = [
        {
            "role": "user",
            "content": prompt
        }
    ]
    response = openai.ChatCompletion.create(
        model=model,
        messages=messages,
        max_tokens=max_tokens,
        stop=stop_strs,
        temperature=temperature,
    )
    return response.choices[0]["message"]["content"]
