import os
from pathlib import Path

import numpy as np
import torch
from huggingface_hub import snapshot_download
from transformers import AutoModelForCausalLM, AutoModelForSeq2SeqLM, AutoTokenizer

from utils.data_reading import read_openapi
from utils.openaiLLM import OpenAIModel


FT5_VARIANT = ["flan-t5-xxl", "flan-t5-xl", "flan-t5-large", "flan-t5-base", "flan-t5-small"]
LLAMA2_VARIANT = ["Llama-2-7b-chat-hf", "Llama-2-13b-chat-hf"]
GPT_VARIANT = ["gpt-5.5"]

FT5_PATH = Path(os.getenv("TELLER_FLAN_T5_DIR", "llms/flanT5"))
LLAMA_PATH = Path(os.getenv("TELLER_LLAMA_DIR", "llms/l2"))
FT5_MODEL_IDS = {name: f"google/{name}" for name in FT5_VARIANT}
LLAMA_MODEL_IDS = {name: f"meta-llama/{name}" for name in LLAMA2_VARIANT}


def _local_or_hub_path(local_root, model_name, model_ids):
    local_path = local_root / model_name
    return str(local_path) if local_path.is_dir() else model_ids[model_name]


class T5_Question_Answering:
    def __init__(self, model_name="flan-t5-small", device="cpu"):
        supported_models = FT5_VARIANT + LLAMA2_VARIANT + GPT_VARIANT
        if model_name not in supported_models:
            raise ValueError(f"Unsupported model: {model_name}")

        self.model_name = model_name
        self.device = "cuda" if device == "auto" and torch.cuda.is_available() else device
        if self.device == "auto":
            self.device = "cpu"

        if model_name in FT5_VARIANT:
            model_path = _local_or_hub_path(FT5_PATH, model_name, FT5_MODEL_IDS)
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
            self.model = AutoModelForSeq2SeqLM.from_pretrained(model_path).to(self.device)
            self.model.eval()
            self.yes_token_id = self.tokenizer.encode("Yes", add_special_tokens=False)[0]
            self.no_token_id = self.tokenizer.encode("No", add_special_tokens=False)[0]
        elif model_name in LLAMA2_VARIANT:
            model_path = _local_or_hub_path(LLAMA_PATH, model_name, LLAMA_MODEL_IDS)
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
            dtype = torch.float32
            if self.device == "cuda" and torch.cuda.is_bf16_supported():
                dtype = torch.bfloat16
            self.model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=dtype).to(self.device)
            self.model.eval()
            self.yes_token_id = self.tokenizer.encode("Yes", add_special_tokens=False)[0]
            self.no_token_id = self.tokenizer.encode("No", add_special_tokens=False)[0]
        else:
            self.model = OpenAIModel(
                API_KEY=read_openapi(),
                model_name=model_name,
                stop_words=[],
                max_new_tokens=3,
                logprobs=None,
                echo=False,
            )

    def generate(self, input_string, **generator_args):
        if self.model_name in FT5_VARIANT + LLAMA2_VARIANT:
            input_ids = self.tokenizer(input_string, return_tensors="pt").input_ids.to(self.device)
            with torch.no_grad():
                output_ids = self.model.generate(input_ids, **generator_args)
            return self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)

        response = self.model.batch_chat_generate(
            [input_string],
            temperature=generator_args.get("temperature", 0.0),
            n=generator_args.get("num_return_sequences", 1),
        )[0]
        return [choice["message"]["content"] for choice in response["choices"]]

    def answer_direct_sampling(self, info, gq, do_sample=True, temperature=1, num_return_sequences=1):
        if info is None:
            input_string = f"{gq}\r\nYes or No? Response:"
        else:
            input_string = f"{info}\r\nBased on the above information, {gq} Yes or No? Response:"

        answers = self.generate(
            input_string,
            max_new_tokens=3,
            do_sample=do_sample,
            temperature=temperature,
            num_return_sequences=num_return_sequences,
        )
        labels = self.map_direct_answer_to_label(answers)
        return float(sum(labels) / len(labels)) if labels else 0.5

    @staticmethod
    def map_direct_answer_to_label(predictions):
        labels = []
        for prediction in predictions:
            normalized = prediction.strip().lower().lstrip("([{\"' ")
            if normalized.startswith("yes"):
                labels.append(1)
            elif normalized.startswith("no"):
                labels.append(0)
        return labels

    def answer_logics(self, info, gq, **kwargs):
        if self.model_name not in FT5_VARIANT + LLAMA2_VARIANT:
            raise ValueError(f"Model {self.model_name} does not support logic scoring.")

        if info is None:
            prompts = [f"{gq} Yes or No? Response:"]
        else:
            prompts = [
                f"{info}\r\n{gq} Yes or No? Response:",
                f"{info}\r\nBased on the above information, {gq} Yes or No? Response:",
            ]

        scores = []
        for prompt in prompts:
            input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids.to(self.device)
            with torch.no_grad():
                if self.model_name in FT5_VARIANT:
                    output = self.model.generate(
                        input_ids,
                        return_dict_in_generate=True,
                        output_scores=True,
                        max_new_tokens=1,
                    )
                    logits = output.scores[0][0]
                else:
                    logits = self.model(input_ids).logits[0, -1]
                yes_no_logits = torch.stack((logits[self.no_token_id], logits[self.yes_token_id]))
                scores.append(torch.softmax(yes_no_logits.float(), dim=0)[1].item())
        return float(np.mean(scores))


def download_T5(path=FT5_PATH):
    for model_name in ("google/flan-t5-small", "google/flan-t5-base"):
        snapshot_download(repo_id=model_name, local_dir=str(Path(path) / model_name.split("/")[-1]))


def download_llama2(path=LLAMA_PATH):
    model_name = "meta-llama/Llama-2-7b-chat-hf"
    snapshot_download(repo_id=model_name, local_dir=str(Path(path) / model_name.split("/")[-1]))


if __name__ == "__main__":
    download_T5()
