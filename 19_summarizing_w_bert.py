import numpy as np
from utils.json_utils import read_json, write_json

sentences_w_subjects_tokenized = read_json("9_non_lemmatized_tokenized_sentences_black_clover.json")

sentences_w_subjects_tokenized = [
    {
        "subjects" : sentence["subjects"],
        "tokens"  : " ".join(sentence["tokens"])
    } 
    for sentence in sentences_w_subjects_tokenized
]

max_length = 0
for sentence in sentences_w_subjects_tokenized:
    if len(sentence["tokens"]) > max_length:
        max_length = len(sentence["tokens"])

from transformers import AutoTokenizer, AutoModelForMaskedLM
import torch

model_checkpoint = "bert-base-cased"

tokenizer = AutoTokenizer.from_pretrained(model_checkpoint, use_fast=True)

model = AutoModelForMaskedLM.from_pretrained(
    model_checkpoint, pad_token_id=tokenizer.eos_token_id)

device = f"cuda:{torch.cuda.current_device()}" if torch.cuda.is_available() else "cpu"

def predict_mask(input_str):
    """Tomamos el camino largo en lugar de usar pipeline
    """
    inputs = tokenizer(input_str, return_tensors="pt")
    mask_index = np.where(inputs['input_ids'] == tokenizer.mask_token_id)
    # .eval() to set dropout and batch normalization layers to evaluation mode
    model.eval()
    outputs = model(**inputs)
    top_5_predictions = torch.softmax(outputs.logits[mask_index], dim=1).topk(5)
    predicted = []
    for i in range(5):
        token = tokenizer.decode(top_5_predictions.indices[0, i])
        prob = top_5_predictions.values[0, i]
        predicted.append({
            "token": token, 
            "prob": prob.item()
        })
    return predicted

predicted_by_sentence_by_character = []
for sentence in sentences_w_subjects_tokenized:
    predicted_by_character = dict()
    for subject in sentence["subjects"]:
        print(sentence["tokens"] + " " + subject + " can be described as [MASK].")
        print()
        predicted = predict_mask(sentence["tokens"] + " " + subject + " can be described as [MASK].")
        predicted_by_character[subject] = predicted
        for i in range(0, len(predicted)):
            print(f" {i+1}) {predicted[i]['token']:<20} {predicted[i]['prob']:.3f}")
        print()
    predicted_by_sentence_by_character.append(predicted_by_character)

write_json(predicted_by_sentence_by_character, "19_summarizing_w_bert.json")