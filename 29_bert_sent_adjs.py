from utils.json_utils import read_json, write_json
from datasets import Dataset
import pandas as pd

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

varied_set_adjectives = read_json("14_varied_set_adjectives_definitions.json")

training_sents = []
for adj in varied_set_adjectives:
    training_sents.append("[MASK] can be described as " + adj + ".")
    training_sents.append("[MASK] can be described as " + adj + ".")
    training_sents.append("[MASK] can be described as " + adj + ".")
    training_sents.append("[MASK] can be described as " + adj + ".")
    training_sents.append("[MASK] can be described as " + adj + ".")

training_sents.extend(list(map(lambda x : x["tokens"], sentences_w_subjects_tokenized)))

from transformers import AutoTokenizer, AutoModelForMaskedLM, DataCollatorForLanguageModeling, Trainer, TrainingArguments
import torch

model_checkpoint = "bert-base-cased"
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint, use_fast=True)

data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)

tokenize_fn = lambda doc : tokenizer(
        doc,
        truncation=True,
        max_length=1024
    )

tokenized_dataset = list(map(tokenize_fn, training_sents))

tokenized_dataset = pd.DataFrame(tokenized_dataset)
tokenized_dataset = Dataset.from_pandas(tokenized_dataset)

model = AutoModelForMaskedLM.from_pretrained(
    model_checkpoint, pad_token_id=tokenizer.eos_token_id)

device = f"cuda:{torch.cuda.current_device()}" if torch.cuda.is_available() else "cpu"

training_args = TrainingArguments(
    f"{model_checkpoint}-finetuned-adjs-wsent-black-clover",
    num_train_epochs=5,
    per_device_train_batch_size=10,
    per_device_eval_batch_size=10,    
    learning_rate=2e-5,
    weight_decay=0.001,
    do_eval=True, # eval en validation set
    evaluation_strategy="steps", # eval en validation set
    eval_steps=5000,
    save_steps=5000, # checkpoint model every 500 steps
    logging_dir='./logs', # logging
    logging_strategy="steps",
    logging_steps=1,
    fp16=False, # float16 en training (only on CUDA)
    push_to_hub=False,
)

trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=tokenized_dataset, #.select(range(0, 128)),
    eval_dataset=tokenized_dataset, #.select(range(0, 128)),
)

train_output = trainer.train()

import numpy as np

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

write_json(predicted_by_sentence_by_character, "29_sent_predicted_by_char_bert_sent_coref_10.json")

predicted_by_sentence_by_character = []
for sentence in sentences_w_subjects_tokenized:
    predicted_by_character = dict()
    for subject in sentence["subjects"]:
        print(subject + " can be described as [MASK].")
        print()
        predicted = predict_mask(subject + " can be described as [MASK].")
        predicted_by_character[subject] = predicted
        for i in range(0, len(predicted)):
            print(f" {i+1}) {predicted[i]['token']:<20} {predicted[i]['prob']:.3f}")
        print()
    predicted_by_sentence_by_character.append(predicted_by_character)

write_json(predicted_by_sentence_by_character, "29_predicted_by_char_bert_sent_coref_10.json")

def predict_mask_x(input_str, x):
    """Tomamos el camino largo en lugar de usar pipeline
    """
    inputs = tokenizer(input_str, return_tensors="pt")
    mask_index = np.where(inputs['input_ids'] == tokenizer.mask_token_id)
    # .eval() to set dropout and batch normalization layers to evaluation mode
    model.eval()
    outputs = model(**inputs)
    top_x_predictions = torch.softmax(outputs.logits[mask_index], dim=1).topk(x)
    predicted = []
    for i in range(x):
        token = tokenizer.decode(top_x_predictions.indices[0, i])
        prob = top_x_predictions.values[0, i]
        predicted.append({
            "token": token, 
            "prob": prob.item()
        })
    return predicted

characters = read_json('3_characters_black_clover.json')
adjectives = read_json("14_varied_set_adjectives.json")

predicted_by_character = dict()
for subject in characters:
    f = open("29_" + subject +"_described_10.txt", "w")
    predicted = predict_mask_x(subject + " can be described as [MASK].", 27)
    predicted_by_character[subject] = predicted

    for i in range(0, len(predicted)):
        to_write = f" {i+1}) {predicted[i]['token']:<20} {predicted[i]['prob']:.3f}\n\n"
        f.write(to_write)
    f.close()

predicted_by_character = dict()
for subject in characters:
    f = open("29_" + subject +"_is_10.txt", "w")
    predicted = predict_mask_x(subject + " is [MASK].", 27)
    predicted_by_character[subject] = predicted

    for i in range(0, len(predicted)):
        to_write = f" {i+1}) {predicted[i]['token']:<20} {predicted[i]['prob']:.3f}\n\n"
        f.write(to_write)
    f.close()

perplex_adjs = dict()
for character in characters:
    perplex_adjs[character] = dict()
    for adj in adjectives:
        sent = character + " is " + adj
        tokenized_sent = tokenizer(sent)
        tokenized_sent = Dataset.from_list([tokenized_sent])
        perplexity = np.exp(trainer.evaluate(tokenized_sent)["eval_loss"])
        perplex_adjs[character][adj] = perplexity

for character in characters:
    sorted_perplex_adjs = sorted(perplex_adjs[character].items(), key=lambda kv: kv[1])
    write_json(sorted_perplex_adjs, "29_" + character + "_is_perplexity_10.json")

perplex_adjs = dict()
for character in characters:
    perplex_adjs[character] = dict()
    for adj in adjectives:
        sent = character + " can be described as " + adj
        tokenized_sent = tokenizer(sent)
        tokenized_sent = Dataset.from_list([tokenized_sent])
        perplexity = np.exp(trainer.evaluate(tokenized_sent)["eval_loss"])
        perplex_adjs[character][adj] = perplexity

for character in characters:
    sorted_perplex_adjs = sorted(perplex_adjs[character].items(), key=lambda kv: kv[1])
    write_json(sorted_perplex_adjs, "29_" + character + "_described_perplexity_10.json")

more_similar_vocab = read_json("14_more_similar_vocabulary.json")

perplex_adjs_simpler = dict()
for character in characters:
    perplex_adjs_simpler[character] = dict()
    for adj in more_similar_vocab:
        sent = character + " is " + adj
        tokenized_sent = tokenizer(sent)
        tokenized_sent = Dataset.from_list([tokenized_sent])
        perplexity = np.exp(trainer.evaluate(tokenized_sent)["eval_loss"])
        perplex_adjs_simpler[character][adj] = perplexity

for character in characters:
    sorted_perplex_adjs = sorted(perplex_adjs_simpler[character].items(), key=lambda kv: kv[1])
    write_json(sorted_perplex_adjs, "29_" + character + "_is_perplexity_simpler_10.json")

perplex_adjs_simpler = dict()
for character in characters:
    perplex_adjs_simpler[character] = dict()
    for adj in more_similar_vocab:
        sent = character + " can be described as " + adj
        tokenized_sent = tokenizer(sent)
        tokenized_sent = Dataset.from_list([tokenized_sent])
        perplexity = np.exp(trainer.evaluate(tokenized_sent)["eval_loss"])
        perplex_adjs_simpler[character][adj] = perplexity

for character in characters:
    sorted_perplex_adjs = sorted(perplex_adjs_simpler[character].items(), key=lambda kv: kv[1])
    write_json(sorted_perplex_adjs, "29_" + character + "_described_perplexity_simpler_10.json")