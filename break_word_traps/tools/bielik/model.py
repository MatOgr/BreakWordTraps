from typing import Optional, List, Dict

import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# device = "cuda"  # the device to load the model onto

model_name = "speakleash/Bielik-7B-v0.1"

TOKENIZER: Optional[AutoTokenizer] = None
MODEL: Optional[AutoModelForCausalLM] = None
PIPELINE: Optional = None


def prepare_model():
    global TOKENIZER, MODEL, PIPELINE
    TOKENIZER = AutoTokenizer.from_pretrained(model_name)
    MODEL = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16)
    PIPELINE = transformers.pipeline(
        "text-generation", model=MODEL, tokenizer=TOKENIZER
    )


def call_model(text: str | List[str], max_token: int = 10):
    # input_ids = TOKENIZER.apply_chat_template(messages, return_tensors="pt")

    # model_inputs = input_ids.to(device)
    # MODEL.to(device)

    # generated_ids = MODEL.generate(model_inputs, max_new_tokens=1000, do_sample=True)
    # return TOKENIZER.batch_decode(generated_ids)
    return PIPELINE(
        max_new_tokens=max_token,
        do_sample=True,
        top_k=10,
        eos_token_id=TOKENIZER.eos_token_id,
        # device="cuda",
        text_inputs=text,
    )


def analyze_text(text: str):
    sentiment = f"{text}\n\nSentyment tego teksu to "
    target_group = f"{text}\n\nText ten jest kierowany do "
    questions = f"{text}\n\n10 pytań dotyczących tego teksu:\n1. "
    # summary = f"{text}\n\nPodsumowując "
    response = call_model([sentiment, target_group])
    response_questions = call_model(questions)
    return {
        "sentiment": response[0]["generated_text"][len(text) + 2 :],
        "targetGroup": response[1]["generated_text"][len(text) + 2 :],
        # "summary": response[3]["generated_text"][len(text) + 2:],
        "questions": response_questions["generated_text"][len(text) + 2 :].split("\n"),
    }


if __name__ == "__main__":
    from pprint import pprint

    prepare_model()
    pprint(
        analyze_text(
            """W pierwszej połowie lipca przeprowadziliśmy ogólnopolską akcję wzmożonej kontroli przesyłek pocztowych oraz kurierskich. Funkcjonariusze przeprowadzili kontrolę w 18 punktach w całej Polsce. Wrześniowa zmiana warunków oprocentowania obligacji oszczędnościowych wynika z potrzeby ich dostosowania do bieżących realiów rynkowych."""
        )
    )
