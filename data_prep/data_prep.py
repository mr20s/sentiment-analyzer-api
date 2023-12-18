from typing import List, Tuple, Optional
import re
import os
import itertools as itt
import functools as fct
import nltk
import string
import pandas as pd
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import MinMaxScaler
import pickle


STOPWORDS_LANG_DICT = {
    "EN": "english",
    "DE": "german",
    "LT": "",
    "IT": "italian",
    "FR": "french",
    "ES": "spanish",
    "NL": "dutch",
    "EL": "greek",
    "PT": "portuguese",
}
LABEL_TRANSLATION_DICT = {0: "negative", 1: "neutral", 2: "positive"}


def compute_punct_percentage(text: str) -> float:
    if not len(text):
        return 0
    text_no_ws = re.sub("\s+", "", text)
    punct = re.sub("\w+|'", "", text_no_ws)
    return len(punct) / len(text_no_ws) * 100


def load_lithuanian_stopwords() -> List[str]:
    lt_stopwords = re.split(
        "\s+", open("./data/lithuanian_stopwords.txt", "r", encoding="UTF-8").read()
    )
    return list(set(lt_stopwords))


def clean_and_tokenize(text: str, stopwords: List[str]) -> List[str]:
    text_nop_low = re.sub(f"[{string.punctuation}]+", "", text).lower()
    text_tokens = re.findall("\w+[€£]|\w+|[\u263a-\U0001f645]+", text_nop_low)
    text_tokens_no_stopwords = [w for w in text_tokens if w not in stopwords]
    return text_tokens_no_stopwords


stopwords_per_country = {
    key: nltk.corpus.stopwords.words(value)
    if key != "LT"
    else load_lithuanian_stopwords()
    for key, value in STOPWORDS_LANG_DICT.items()
}
all_stopwords = list(itt.chain.from_iterable(stopwords_per_country.values()))


def data_preparation(dataset_path: str, save: bool = False) -> "pd.DataFrame":
    assert os.path.exists(
        dataset_path
    ), f"[data_preparation] {dataset_path} does not exist!"

    data = pd.read_csv(dataset_path, encoding="utf-8")
    data.review_content = data.review_content.fillna("")
    data = data.assign(
        review_content_tokenized=data.review_content.map(
            fct.partial(clean_and_tokenize, stopwords=all_stopwords), na_action="ignore"
        ),
        review_content_tokenized_str=lambda df: df.review_content_tokenized.map(lambda el: " ".join(el)),
        sentiment=data.overall_rating.replace(
            to_replace={5: "pos", 4: "pos", 3: "neu", 2: "neg", 1: "neg"}
        ),
    )

    # adding other features
    data = data.assign(
        review_length_tokens=data.review_content_tokenized.apply(
            len
        ),  # length in terms of significant tokens
        review_punct_perc=lambda df: df.review_content.apply(
            compute_punct_percentage
        ),  # % of punctuation in review
    )

    if save:
        save_path = dataset_path[
            :-4
        ]  # for simplicity, only .csv are supported in this prototype
        save_path += "_processed.csv"
        data.to_csv(f"{save_path}", index=False)

    return data


def make_data_for_model(data: "pd.DataFrame") -> Tuple["pd.DataFrame", int]:
    all_words = set(itt.chain.from_iterable(data.review_content_tokenized.tolist()))
    num_words = len(all_words)
    max_review_significant_length = data.review_content_tokenized_str.apply(len).max()

    # tokenizing sentences and labelling words
    tokenizer = Tokenizer(num_words=num_words, oov_token="<OOV>")
    tokenizer.fit_on_texts(data["review_content"])
    word_index = tokenizer.word_index  # to save
    sequences = tokenizer.texts_to_sequences(data["review_content"])
    padded_sequences = pad_sequences(
        sequences, maxlen=max_review_significant_length, truncating="post"
    )
    # creating labels
    labels = pd.get_dummies(data["sentiment"])

    # retrieving previosly generated features
    generated_numeric_features = data[
        ["review_length_tokens", "review_punct_perc"]
    ].copy()

    # scaling data
    rev_len_scaler = MinMaxScaler().fit(
        generated_numeric_features.review_length_tokens.values[:, np.newaxis]
    )
    punct_perc_scaler = MinMaxScaler().fit(
        generated_numeric_features.review_punct_perc.values[:, np.newaxis]
    )
    # adding scaled features
    generated_numeric_features = generated_numeric_features.assign(
        review_length_tokens_normed=rev_len_scaler.transform(
            generated_numeric_features.review_length_tokens.values[:, np.newaxis]
        ),
        review_punct_perc_normed=punct_perc_scaler.transform(
            generated_numeric_features.review_punct_perc.values[:, np.newaxis]
        ),
    )
    normed_numeric_features = generated_numeric_features[
        ["review_length_tokens_normed", "review_punct_perc_normed"]
    ]

    # computing number of embedding dimentions
    # num_embedding_out_dims = round(np.sqrt(num_words))

    full_dataset = pd.DataFrame(
        data=np.hstack(
            (padded_sequences, normed_numeric_features.values, labels.values)
        ),
        columns=[
            *np.arange(padded_sequences.shape[1]),
            *normed_numeric_features.columns,
            *labels.columns,
        ],
        index=data.index,
    )

    # pickling tokenizer and scalers
    with open("./models/support_objects/tf_tokenizer.pickle", "wb") as handle:
        pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open("./models/support_objects/word_index.pickle", "wb") as handle:
        pickle.dump(word_index, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open("./models/support_objects/rev_len_scaler.pickle", "wb") as handle:
        pickle.dump(rev_len_scaler, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open("./models/support_objects/punct_perc_scaler.pickle", "wb") as handle:
        pickle.dump(punct_perc_scaler, handle, protocol=pickle.HIGHEST_PROTOCOL)

    return full_dataset, len(word_index)


def split_train_test_validation_sets(
    full_dataset: "pd.DataFrame",
    validation_percentage: float = 0.05,
    test_percentage: float = 0.15,
    save_path: Optional[str] = None,
    return_test_set: bool = False,
) -> Tuple["pd.DataFrame"]:
    train_sets = []
    valid_sets = []
    test_sets = []

    for label_col in full_dataset.columns[-3:]:  # negative, neutral, positive
        sub_dataset = full_dataset.query(f"{label_col} == 1")
        curr_label_val_set = sub_dataset.sample(frac=validation_percentage).sort_index()

        curr_label_train_test_set = sub_dataset.drop(index=curr_label_val_set.index)
        curr_label_test_set = curr_label_train_test_set.sample(
            frac=test_percentage
        ).sort_index()

        curr_label_train_set = curr_label_train_test_set.drop(
            index=curr_label_test_set.index
        )

        train_sets.append(curr_label_train_set)
        valid_sets.append(curr_label_val_set)
        test_sets.append(curr_label_test_set)

    train_set = pd.concat(train_sets).sort_index()
    validation_set = pd.concat(valid_sets).sort_index()
    test_set = pd.concat(test_sets).sort_index()

    if save_path:
        save_path_no_ext = save_path[
            :-4
        ]  # for simplicity, only .csv are supported in this prototype
        train_save_path = save_path_no_ext + "_train_set.csv"
        valid_save_path = save_path_no_ext + "_valid_set.csv"
        test_save_path = save_path_no_ext + "_test_set.csv"
        train_set.to_csv(train_save_path)
        validation_set.to_csv(valid_save_path)
        test_set.to_csv(test_save_path)

    if return_test_set:
        return train_set, validation_set, test_set
    return train_set, validation_set


def prepare_input_sentence_to_predict(
    text: str, max_review_significant_length: int
) -> List[List["Numeric"]]:
    # loading tokenizer and scalers
    with open("./models/support_objects/tf_tokenizer.pickle", "rb") as handle:
        tokenizer = pickle.load(handle)
    with open("./models/support_objects/rev_len_scaler.pickle", "rb") as handle:
        rev_len_scaler = pickle.load(handle)
    with open("./models/support_objects/punct_perc_scaler.pickle", "rb") as handle:
        punct_perc_scaler = pickle.load(handle)

    # tokenizing sentences and labelling words
    sequence = tokenizer.texts_to_sequences([text])
    padded_sequence = pad_sequences(
        sequence, maxlen=max_review_significant_length, truncating="post"
    )

    # extracting numeric features
    text_tokens_length = len(clean_and_tokenize(text, all_stopwords))
    punct_perc = compute_punct_percentage(text)
    text_tokens_length_normed = rev_len_scaler.transform([[text_tokens_length]])
    punct_perc_normed = punct_perc_scaler.transform([[punct_perc]])

    return [
        padded_sequence.astype(np.float64),
        np.hstack((text_tokens_length_normed, punct_perc_normed)).astype(np.float64),
    ]
