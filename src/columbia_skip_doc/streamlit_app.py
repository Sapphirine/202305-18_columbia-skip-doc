"""
Streamlit main script
"""

import argparse
import logging
import sys
import os
import torch

import streamlit as st

from utils import setup_logging
from streamlit_chat import message
from constants import PRJ_ROOT_STR
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
)

__author__ = "Charles Antoine Malenfant & Lance Norman"
__copyright__ = "Charles Antoine Malenfant & Lance Norman"
__license__ = "MIT"

CLASS_NAME = __name__

symptom, where, when, what, duration, pain, worse, improve, effect = (
    " ",
    " ",
    " ",
    " ",
    " ",
    " ",
    " ",
    " ",
    " ",
)


def parse_args(args):
    """Parse command line parameters

    Args:
      args (List[str]): command line parameters as list of strings
          (for example  ``["--help"]``).

    Returns:
      :obj:`argparse.Namespace`: command line parameters namespace
    """
    parser = argparse.ArgumentParser(description="Streamlit app launch")

    parser.add_argument(
        "-v",
        "--verbose",
        dest="loglevel",
        help="set loglevel to INFO",
        action="store_const",
        const=logging.INFO,
    )
    parser.add_argument(
        "-vv",
        "--very-verbose",
        dest="loglevel",
        help="set loglevel to DEBUG",
        action="store_const",
        const=logging.DEBUG,
    )

    parser.add_argument("--use_cuda", action="store_true")

    return parser.parse_args(args)


def get_input():
    input_text = st.text_input("Patient (You)", key="input")
    return input_text


def loading():
    st.write("Loading...")


def symptoms(widget):
    st.session_state["symptom"] = widget.selectbox(
        "Select the character of your symptoms",
        (
            "Sharp pain",
            "Dull pain",
            "Irritation",
            "Discomfort",
            "Fatigue",
            "Nausea",
            "Cough",
            "Fever",
            "Shortness of breath",
            "Other",
        ),
    )


def location(widget):
    st.session_state["where"] = widget.selectbox(
        "Where are those symptoms located?",
        (
            "Head",
            "Limbs",
            "Chest",
            "Abdomen",
            "Pelvic Area",
            "Systemic",
        ),
    )


def when_start(widget):
    st.session_state["when"] = widget.selectbox(
        "When did this start happening?",
        (
            "Over a year ago",
            "6 months to a year ago",
            "3 to 6 months ago",
            "1 to 3 months ago",
            "2 to 4 weeks ago",
            "Within the last week",
            "1 to 3 days ago",
            "Within the last day",
            "Within the last 12 hours",
            "Within the last 6 hours",
            "1 hour ago or less",
        ),
    )


def what_doing(widget):
    st.session_state["what"] = widget.selectbox(
        "What were you doing when this started?",
        (
            "At rest",
            "During physical exertion",
        ),
    )


def how_long(widget):
    st.session_state["duration"] = widget.selectbox(
        "Please describe the duration of your symptoms.",
        (
            "Constant",
            "Comes and goes",
        ),
    )


def severity(widget):
    st.session_state["pain"] = widget.slider(
        "On a scale of 1 to 10, with 1 being very mild and 10 being the worst of your life, please rate the intensity of your symptoms.",
        min_value=1,
        max_value=10,
    )


def factors_worse(widget):
    st.session_state["worse"] = widget.selectbox(
        "What factors make your symptoms worse?",
        (
            "Physical exertion",
            "Cold Showers",
            "Hot Showers",
            "Being outside",
            "Eating certain foods",
            "Skin contact",
            "Loud sounds",
            "Exposure to light",
            "Daily activities",
        ),
    )


def factors_better(widget):
    st.session_state["improve"] = widget.selectbox(
        "What factors make your symptoms better?",
        (
            "Cold Showers",
            "Hot Showers",
            "Taking medication",
            "Rest",
        ),
    )


def affect(widget):
    st.session_state["effect"] = widget.selectbox(
        "How are these symptoms affecting your life?",
        (
            "Unable to go work/school",
            "Unable to sleep",
            "Unable to partake in social activities",
            "Attending the above but with compromised performance",
        ),
    )


def increment_counter():
    st.session_state.count += 1


def restart():
    st.session_state.count = 1


@st.cache_data
def load_model(use_cuda):
    device = torch.device("cuda" if use_cuda else "cpu")
    models_fp = os.path.join(PRJ_ROOT_STR, "src/columbia_skip_doc/models")
    model = AutoModelForCausalLM.from_pretrained(models_fp)
    # model = AutoModelForCausalLM.from_pretrained("gpt2")
    model.to(device)
    model.eval()
    return model


# Set to use cuda


def main(args):
    """Wrapper allowing

    Args:
      args (List[str]): command line parameters as list of strings
          (for example  ``["--verbose", "42"]``).
    """

    # parser = argparse.ArgumentParser("")
    args = parse_args(args)
    use_cuda = args.use_cuda and torch.cuda.is_available()
    _logger = setup_logging(logging.DEBUG, CLASS_NAME)

    if "count" not in st.session_state:
        st.session_state.count = 0

    st.title("Skip-DOC")
    if use_cuda:
        max_length = 500
        st.write("Model loaded with CUDA, max response length = ", max_length)
    else:
        max_length = 100
        st.write("Model loaded on CPU, max response length = ", max_length)

    subheader_placeholder = st.empty()
    subheader_placeholder = subheader_placeholder.subheader(
        "Please answer each question with a choice from the drop-down menu", anchor=None
    )
    if "generated" not in st.session_state:
        st.session_state["generated"] = []
    if "past" not in st.session_state:
        st.session_state["past"] = []

    placeholder = st.empty()
    button_placeholder = st.empty()
    button_placeholder.button("Press to Continue", on_click=increment_counter)

    if st.session_state.count == 1:
        symptoms(placeholder)
    elif st.session_state.count == 2:
        location(placeholder)
    elif st.session_state.count == 3:
        when_start(placeholder)
    elif st.session_state.count == 4:
        what_doing(placeholder)
    elif st.session_state.count == 5:
        how_long(placeholder)
    elif st.session_state.count == 6:
        severity(placeholder)
    elif st.session_state.count == 7:
        factors_worse(placeholder)
    elif st.session_state.count == 8:
        factors_better(placeholder)
    elif st.session_state.count == 9:
        affect(placeholder)
    elif st.session_state.count == 10:
        button_placeholder.button("Please Wait", disabled=True)
        st.session_state.count = 0

        subheader_placeholder = subheader_placeholder.subheader(
            "Please wait while Skip-Doc generates a response.",
            anchor=None,
        )

        input_text = "".join(
            [
                "The patient is showing signs of ",
                str(st.session_state.symptom).lower(),
                " in the ",
                str(st.session_state.where).lower(),
                " that started ",
                str(st.session_state.when).lower(),
                " while ",
                str(st.session_state.what).lower(),
                " and has been ",
                str(st.session_state.duration).lower(),
                ". The patient rated the severity of the symptoms as ",
                str(st.session_state.pain).lower(),
                " out of 10. The symptoms are made worse by ",
                str(st.session_state.worse).lower(),
                " and improved by ",
                str(st.session_state.improve).lower(),
                ". The symptoms are causing the patient to be ",
                str(st.session_state.effect).lower(),
                ".",
            ]
        )

        generation_arguments = {
            "max_new_tokens": max_length,
            "min_length": 5,
            "temperature": 1.0,
            "do_sample": True,
            "top_k": 0,
            "top_p": 0.9,
            "repetition_penalty": 6.0,
            "num_beams": 2,
        }

        input_text = (
            input_text
            + ". You are a doctor in a clinic. Answer the question and provide a plan of action if needed."
        )

        tokenizer = AutoTokenizer.from_pretrained("gpt2")

        inputs = tokenizer(
            input_text,
            return_tensors="pt",
        )

        device = torch.device("cuda" if use_cuda else "cpu")
        inputs.to(device)

        model = load_model(use_cuda)

        output = model.generate(**inputs, **generation_arguments)
        generated_sentence = tokenizer.decode(output[0], skip_special_tokens=True)
        output_sentence = generated_sentence.replace(input_text, "")
        output_sentence = output_sentence.replace("Question: ", "")
        output_sentence = output_sentence.replace(
            ". You are a doctor in a clinic. Answer the question and provide a plan of action if needed.",
            "",
        )

        input_text = input_text.replace("Question: ", "")
        input_text = input_text.replace(
            ". You are a doctor in a clinic. Answer the question and provide a plan of action if needed.",
            "",
        )

        output_sentence.replace("\n", " ")
        output_sentence.strip()

        st.write("".join(["Patient's Condition: \n\r", input_text]))

        st.write("".join(["Skip-Doc's Reply: \n\r", output_sentence]))
        print("Output:\n" + 100 * "-")
        print(output_sentence)

        button_placeholder.button("Restart", on_click=restart)

    _logger.info("Script ends here")


def run():
    """Calls :func:`main` passing the CLI arguments extracted from :obj:`sys.argv`

    This function can be used as entry point to create console scripts with setuptools.
    """
    main(sys.argv[1:])


if __name__ == "__main__":
    run()
