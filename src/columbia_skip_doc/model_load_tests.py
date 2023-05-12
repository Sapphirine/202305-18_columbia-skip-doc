"""
Test script for testing pretrained OpenPrompt models on sample text
"""

import logging
import os
import torch
import argparse
import sys

from utils import setup_logging
from constants import PRJ_ROOT_STR

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
)

__author__ = "Charles Antoine Malenfant & Lance Norman"
__copyright__ = "Charles Antoine Malenfant & Lance Norman"
__license__ = "MIT"

CLASS_NAME = __name__
FILE_NAME = __name__
_logger = setup_logging(logging.DEBUG, FILE_NAME)


if __name__ == "__main__":
    parser = argparse.ArgumentParser("")
    parser.add_argument("--use_cuda", type=bool, default=True)
    args = parser.parse_args()
    use_cuda = args.use_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    models_fp = os.path.join(PRJ_ROOT_STR, "src/columbia_skip_doc/models")
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    model = AutoModelForCausalLM.from_pretrained(models_fp)
    # model = AutoModelForCausalLM.from_pretrained("gpt2")
    model.to(device)
    model.eval()

    # Modify for better responses
    generation_arguments = {
        "max_new_tokens": 200,
        "min_length": 5,
        "temperature": 1.1,
        "do_sample": True,
        "top_k": 0,
        "top_p": 0.9,
        "repetition_penalty": 6.0,
        "num_beams": 2,
    }

    user_input = "What are the complications of Anaplastic thyroid cancer? (Also called: Anaplastic carcinoma of the thyroid)"
    input_text = (
        "Question: "
        + user_input
        + ". You are a doctor in a clinic. Answer the question and provide a plan of action if needed."
    )

    inputs = tokenizer(
        input_text,
        return_tensors="pt",
    )
    # inputs.to(device)

    # torch.manual_seed(100)

    output = model.generate(**inputs, **generation_arguments)
    generated_sentence = tokenizer.decode(output[0], skip_special_tokens=True)
    output_sentence = generated_sentence.replace(user_input, "")
    output_sentence = output_sentence.replace("Question: ", "")
    output_sentence = output_sentence.replace(
        ". You are a doctor in a clinic. Answer the question and provide a plan of action if needed.",
        "",
    )
    output_sentence.replace("\n", " ")
    output_sentence.strip()
    print(input_text + "\r\n")
    print("Output:\n" + 100 * "-")
    print(output_sentence)
