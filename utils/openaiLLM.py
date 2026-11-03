"""Small compatibility wrapper for OpenAI's current Python SDK."""

import os

import tiktoken
from openai import OpenAI

from utils.data_reading import read_openapi


LEGACY_COMPLETION_MODELS = {
    "text-davinci-002",
    "text-davinci-003",
    "code-davinci-002",
}


class OpenAIModel:
    def __init__(self, API_KEY, model_name, stop_words, max_new_tokens, logprobs=None, echo=False):
        api_key = API_KEY if API_KEY and API_KEY != "default" else read_openapi()
        self.client = OpenAI(api_key=api_key)
        self.model_name = model_name
        self.chat_model_name = os.getenv("TELLER_OPENAI_MODEL", "gpt-5.5")
        self.max_new_tokens = max_new_tokens
        self.stop_words = stop_words or None
        self.logprobs = logprobs
        self.echo = echo

    def _chat(self, input_string, temperature=0.0, n=1):
        model = self.chat_model_name if self.model_name in LEGACY_COMPLETION_MODELS else self.model_name
        kwargs = {
            "model": model,
            "messages": [{"role": "user", "content": input_string}],
            "max_completion_tokens": self.max_new_tokens,
            "n": n,
        }
        if self.stop_words:
            kwargs["stop"] = self.stop_words
        # Current reasoning models may reject a custom temperature.
        if not model.startswith("gpt-5") and not model.startswith("o"):
            kwargs["temperature"] = temperature
        return self.client.chat.completions.create(**kwargs)

    @staticmethod
    def _legacy_response(response):
        """Expose dict fields used by the research scripts while using SDK objects."""
        result = response.model_dump()
        for choice in result.get("choices", []):
            content = choice.get("message", {}).get("content") or ""
            choice["text"] = content
        return result

    def chat_generate(self, input_string, temperature=0.0, n=1):
        return self._legacy_response(self._chat(input_string, temperature=temperature, n=n))

    def prompt_generate(self, input_string, temperature=0.0):
        response = self.chat_generate(input_string, temperature=temperature)
        return response["choices"][0]["text"].strip()

    def batch_chat_generate(self, messages_list, temperature=0.0, n=1):
        return [self.chat_generate(prompt, temperature=temperature, n=n) for prompt in messages_list]

    def batch_prompt_generate(self, prompt_list, temperature=0.0):
        return [self.prompt_generate(prompt, temperature=temperature) for prompt in prompt_list]

    def batch_generate(self, messages_list, temperature=0.0):
        return self.batch_chat_generate(messages_list, temperature=temperature)

    def generate(self, input_string, temperature=0.0):
        return self.chat_generate(input_string, temperature=temperature)

    def generate_insertion(self, input_string, suffix, temperature=0.0):
        raise NotImplementedError(
            "The legacy text-completion insertion endpoint is not supported by this wrapper. "
            "Use a chat-completions model instead."
        )

    def token_counting(self, text):
        if self.model_name in LEGACY_COMPLETION_MODELS:
            encoding = tiktoken.get_encoding("p50k_base")
        elif self.model_name == "davinci":
            encoding = tiktoken.get_encoding("gpt2")
        else:
            encoding = tiktoken.get_encoding("cl100k_base")
        return encoding.encode(text)
