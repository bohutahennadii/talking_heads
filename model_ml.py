from transformers import Trainer, TrainingArguments, T5ForConditionalGeneration, T5Tokenizer, AutoModelForCausalLM, AutoTokenizer
from keybert import KeyBERT
from difflib import get_close_matches
import spacy
import json
import re
import random
from spacy.lang.en import English
import torch
from torch.utils.data import Dataset

class ResponseModel(object):
    result: str

nlp = spacy.load("en_core_web_sm")
sent_nlp = English()
sent_nlp.add_pipe("sentencizer")

# Завантаження моделей
kw_model = KeyBERT('sentence-transformers/all-MiniLM-L6-v2')
t5_tokenizer = T5Tokenizer.from_pretrained("t5-small")
t5_model = T5ForConditionalGeneration.from_pretrained("t5-small")
dialo_tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-small")
dialo_model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-small")

# Константи

FILE_PATH = "./models/test-data/text_monument_to_taras_shevchenko.txt"
MIN_RESPONSE_LENGTH = 3
MAX_HISTORY_LENGTH = 500
SYNONYMS = {
    'monument': ['memorial', 'statue', 'sculpture'],
    'memorial': ['monument', 'statue', 'sculpture']
}

# Кеш бази знань
knowledge_base_cache = {}
dialog_history = []

def clean_text(text):
    text = re.sub(r"\[.*?\]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def expand_synonyms(text):
    words = text.lower().split()
    expanded = []
    for word in words:
        expanded.append(word)
        if word in SYNONYMS:
            expanded.extend(SYNONYMS[word])
    return ' '.join(list(set(expanded)))

def lemmatize_text(text):
    doc = nlp(text)
    lemmas = []
    for token in doc:
        if token.pos_ in ["PROPN", "NOUN"]:
            lemmas.append(token.text.lower())
        else:
            lemmas.append(token.lemma_.lower())
    return ' '.join(lemmas)

def extract_keywords(text):
    doc = nlp(text)

    # KeyBERT keywords
    keywords = kw_model.extract_keywords(text, keyphrase_ngram_range=(1, 3), top_n=10)

    # Іменовані сутності
    ner_keywords = [ent.text.lower() for ent in doc.ents]

    # Синтаксично важливі слова
    syntax_keywords = [
        token.lemma_.lower()
        for token in doc
        if token.dep_ in ("ROOT", "nsubj", "dobj") and not token.is_stop
    ]

    # Числові значення
    numeric_values = [token.text for token in doc if token.like_num]

    return list(set(
        [kw[0].lower() for kw in keywords] +
        ner_keywords +
        syntax_keywords +
        numeric_values
    ))

def load_knowledge_base(file_path):
    if file_path in knowledge_base_cache:
        return knowledge_base_cache[file_path]

    knowledge_base = {}
    with open(file_path, "r", encoding="utf-8") as file:
        text = clean_text(file.read())
        doc = sent_nlp(text)
        sentences = [sent.text.strip() for sent in doc.sents]

        for sentence in sentences:
            if not sentence:
                continue

            sentence_clean = lemmatize_text(sentence)
            keywords = extract_keywords(sentence_clean)

            for keyword in keywords:
                if keyword not in knowledge_base or len(sentence) > len(knowledge_base[keyword]):
                    knowledge_base[keyword] = sentence.strip()

    knowledge_base_cache[file_path] = knowledge_base
    return knowledge_base

def generate_t5_response(prompt):
    input_ids = t5_tokenizer(prompt, return_tensors="pt").input_ids
    outputs = t5_model.generate(
        input_ids,
        max_length=100,
        num_beams=5,
        early_stopping=True,
        no_repeat_ngram_size=2,
        repetition_penalty=1.2
    )
    return t5_tokenizer.decode(outputs[0], skip_special_tokens=True)

def generate_dialogpt_response(user_input):
    input_ids = dialo_tokenizer.encode(
        user_input + dialo_tokenizer.eos_token,
        return_tensors='pt'
    )
    outputs = dialo_model.generate(
        input_ids,
        max_length=200,
        pad_token_id=dialo_tokenizer.eos_token_id,
        num_beams=3,
        no_repeat_ngram_size=2
    )
    return dialo_tokenizer.decode(outputs[:, input_ids.shape[-1]:][0], skip_special_tokens=True)

def is_valid_response(response, question):
    response_lower = response.lower()
    question_lower = question.lower()

    if len(response.split()) < MIN_RESPONSE_LENGTH:
        return False

    if any(response_lower.startswith(w) for w in ["who", "when", "where", "what", "how"]):
        return False

    if "?" in response:
        return False

    # Перевірка на повторення питання
    if any(word in response_lower.split() for word in question_lower.split()[:3]):
        return False

    return True

def find_direct_answer(context, question):
    doc = nlp(context)
    q_type = None

    if 'who' in question.lower():
        q_type = 'PERSON'
    elif 'when' in question.lower():
        q_type = 'DATE'
    elif 'where' in question.lower():
        q_type = 'GPE'

    if q_type:
        for ent in doc.ents:
            if ent.label_ == q_type:
                return ent.text
    return None

def get_response(user_input):
    user_input_clean = clean_text(user_input)
    user_input_expanded = expand_synonyms(user_input_clean)
    user_input_lem = lemmatize_text(user_input_expanded)
    keywords = extract_keywords(user_input_lem)
    knowledge_base = load_knowledge_base(FILE_PATH)

    best_answer = None
    used_context = None

    # Пошук у базі знань
    for keyword in keywords:
        if keyword in knowledge_base:
            context = knowledge_base[keyword]
            direct_answer = find_direct_answer(context, user_input_clean)

            if direct_answer:
                best_answer = direct_answer
                used_context = context
                break

            prompt = f"Answer this question: {user_input_clean} using context: {context}"
            response = generate_t5_response(prompt)

            if is_valid_response(response, user_input_clean):
                best_answer = response
                used_context = context
                break

    # Пошук схожих ключових слів
    if not best_answer:
        possible_matches = get_close_matches(
            user_input_lem.lower(),
            knowledge_base.keys(),
            n=2,
            cutoff=0.6
        )
        for match in possible_matches:
            context = knowledge_base[match]
            direct_answer = find_direct_answer(context, user_input_clean)

            if direct_answer:
                best_answer = direct_answer
                used_context = context
                break

            prompt = f"Answer: {user_input_clean} Context: {context}"
            response = generate_t5_response(prompt)

            if is_valid_response(response, user_input_clean):
                best_answer = response
                used_context = context
                break

    # Резервна генерація
    if not best_answer:
        response = generate_dialogpt_response(user_input_clean)
        if not is_valid_response(response, user_input_clean):
            response = "I need more information to answer that. Could you please rephrase your question?"
        best_answer = response

    # Логування
    print(f"Debug - Used context: {used_context}")  # Для налагодження

    # Оновлення історії
    dialog_history.append({"user": user_input, "bot": best_answer})
    if len(dialog_history) > MAX_HISTORY_LENGTH:
        dialog_history.pop(0)

    return best_answer

def save_dialog_history():
    with open("dialog_history.json", "w", encoding="utf-8") as f:
        json.dump(dialog_history[-50:], f, indent=4)

def get_answer(user_input):
    knowledge_base = load_knowledge_base(FILE_PATH)
    response = get_response(user_input)
    print(f"Bot: {response}")
    responseModel = ResponseModel()
    responseModel.result = response

    return responseModel
