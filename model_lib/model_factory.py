import os
import sys
import json
from typing import Optional, Union, Dict
import numpy as np
import pandas as pd
import tensorflow as tf
from keras.models import load_model
from keras.callbacks import EarlyStopping
from sklearn.metrics import accuracy_score

from .sentiment_analyzer import SentimentAnalyzer
from data_prep import (
    data_preparation,
    make_data_for_model,
    split_train_test_validation_sets,
    prepare_input_sentence_to_predict,
    LABEL_TRANSLATION_DICT,
)


class ModelFactory:
    _instance = None

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = ModelFactory()
        return cls._instance

    @property
    def train_set_tuple(self):
        return self._current_train_set_tuple

    @property
    def validation_set_tuple(self):
        return self._current_validation_set_tuple

    @property
    def test_set_tuple(self):
        return self._current_test_set_tuple

    def __init__(self):
        if ModelFactory._instance is not None:
            raise Exception("ModelFactory already exists")
        super().__init__()
        self._models = {}

        # the following can be implemented as NamedTuple
        self._current_train_set_tuple = None
        self._current_validation_set_tuple = None
        self._current_test_set_tuple = None

    def get_loadable_model_names(self):
        model_names_list = [
            name.split(os.path.sep)[-1].split(".")[0]
            for name in os.listdir("./models")
            if name.endswith(".tf")
        ]
        return model_names_list

    def prepare_current_data(
        self,
        data_path: str = "./data/dataset.csv",
    ) -> Dict[str, int]:
        # load and preprocess data - it could be better to have a DataFactory with this method, to share responsibilities
        prepared_data = data_preparation(data_path)
        max_review_significant_length = (
            prepared_data.review_content_tokenized_str.apply(len).max()
        )
        dataset, word_index_length = make_data_for_model(prepared_data)
        train_set, valid_set, test_set = split_train_test_validation_sets(
            dataset, save_path=data_path, return_test_set=True
        )
        # FIXME: smarter ways to do the following splits can be implemented
        # The following tuples contain: (word data, numeric data, labels)
        train_set_tuple = (
            train_set.iloc[:, :max_review_significant_length].values,
            train_set.iloc[
                :, max_review_significant_length : (max_review_significant_length + 2)
            ].values,
            train_set.iloc[:, -3:].values,
        )
        valid_set_tuple = (
            valid_set.iloc[:, :max_review_significant_length].values,
            valid_set.iloc[
                :, max_review_significant_length : (max_review_significant_length + 2)
            ].values,
            valid_set.iloc[:, -3:].values,
        )
        test_set_tuple = (
            test_set.iloc[:, :max_review_significant_length].values,
            test_set.iloc[
                :, max_review_significant_length : (max_review_significant_length + 2)
            ].values,
            test_set.iloc[:, -3:].values,
        )

        self._current_train_set_tuple = train_set_tuple
        self._current_validation_set_tuple = valid_set_tuple
        self._current_test_set_tuple = test_set_tuple

        return {
            "max_review_significant_length": int(max_review_significant_length),
            "word_index_length": int(word_index_length),
        }

    def create_and_train_new_model(
        self,
        max_review_significant_length: int,
        word_index_length: int,
        embedding_out_dims: Optional[int] = None,
        name: str = "sentimentANN",
        save_model: bool = True,
        epochs: int = 10,
        batch_size: int = 128,
        **kwargs,
    ) -> "tf.keras.models.Model":
        if name in self._models:
            print(
                f"[ModelFactory.create_and_train_new_model] - model named {name} already exist. Returning it. "
                "If you want to create a new one, give it a different name or delete the existing one."
            )
            return self._models[name]

        # load and preprocess data - it could be better to have a DataFactory too
        # prepared_data = data_preparation(data_path)
        # max_review_significant_length = (
        #     prepared_data.review_content_tokenized_str.apply(len).max()
        # )
        # dataset, word_index_length = make_data_for_model(prepared_data)
        # train_set, valid_set, test_set = split_train_test_validation_sets(
        #     dataset, save_path=data_path, return_test_set=True
        # )

        if not embedding_out_dims:
            embedding_out_dims = np.sqrt(word_index_length)

        if not self._current_train_set_tuple:
            raise RuntimeError(
                "[ModelFactory.create_and_train_new_model] A dataset must be loaded. Invoke the prepare_data method. "
                "The current training set is None."
            )
        if not self._current_validation_set_tuple:
            raise RuntimeError(
                "[ModelFactory.create_and_train_new_model] A dataset must be loaded. Invoke the prepare_data method. "
                "The current validation set is None."
            )

        (
            train_set_word_ary,
            train_set_numeric_ary,
            train_labels,
        ) = self._current_train_set_tuple
        (
            valid_set_word_ary,
            valid_set_numeric_ary,
            valid_labels,
        ) = self._current_validation_set_tuple

        # computing class weights (NOTE: these may be optional, and driven by input parameters)
        class_weight = (
            pd.Series(data=(len(train_labels) / (3 * train_labels.sum(axis=0))))
            .replace({np.inf: 0.0})
            .reset_index(drop=True)
            .to_dict()
        )

        # smarter ways to do the following splits can be implemented
        # (train_set_word_ary, train_set_numeric_ary, train_labels) = (
        #     train_set.iloc[:, :max_review_significant_length].values,
        #     train_set.iloc[
        #         :, max_review_significant_length : (max_review_significant_length + 2)
        #     ].values,
        #     train_set.iloc[:, -3:].values,
        # )
        # (valid_set_word_ary, valid_set_numeric_ary, valid_labels) = (
        #     valid_set.iloc[:, :max_review_significant_length].values,
        #     valid_set.iloc[
        #         :, max_review_significant_length : (max_review_significant_length + 2)
        #     ].values,
        #     valid_set.iloc[:, -3:].values,
        # )
        # (test_set_word_ary, test_set_numeric_ary, test_labels) = (
        #     test_set.iloc[:, :max_review_significant_length].values,
        #     test_set.iloc[
        #         :, max_review_significant_length : (max_review_significant_length + 2)
        #     ].values,
        #     test_set.iloc[:, -3:].values,
        # )

        # create new model and add to self._models dict
        model = SentimentAnalyzer(
            max_review_significant_length,
            word_index_length,
            embedding_out_dims,
            **kwargs,
        )
        model.compile(
            loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"]
        )

        # training model
        early_stopping = EarlyStopping(
            patience=int(0.1 * epochs), restore_best_weights=True
        )
        _ = model.fit(
            x=[train_set_word_ary, train_set_numeric_ary],
            y=train_labels,
            batch_size=batch_size,
            epochs=epochs,
            validation_data=(
                [valid_set_word_ary, valid_set_numeric_ary],
                valid_labels,
            ),
            shuffle=True,
            class_weight=class_weight,
            callbacks=[early_stopping],
        )

        self._models[name] = model

        if save_model:
            save_model_path = os.path.join(
                "./models", f"{name}.tf"
            )  # should improve overall paths handling
            model.save(save_model_path, overwrite=False)  # could use onnx

        return model

    def test_model_accuracy(self, model_name: str) -> float:
        if not self._current_test_set_tuple:
            raise RuntimeError(
                "[ModelFactory.test_model_accuracy] A dataset must be loaded. Invoke the prepare_data method. "
                "The current test set is None."
            )
        if not model_name in self._models:
            # try to load the model
            self._models[model_name] = self.load_model(model_name)

        # if model still not available return empty sentiment
        if not model_name in self._models:
            print(
                f"[ModelFactory.predict_input_with_specific_model] - model named {model_name} doesn't exist. Returning empty sentiment.",
                file=sys.stderr,
            )
            return np.nan

        (
            test_set_word_ary,
            test_set_numeric_ary,
            test_labels,
        ) = self._current_test_set_tuple
        predictions = self._models[model_name].predict(
            [test_set_word_ary, test_set_numeric_ary]
        )
        predictions_cat = np.argmax(predictions, axis=1)
        test_labels_cat = np.argmax(test_labels, axis=1)
        accuracy = accuracy_score(test_labels_cat, predictions_cat)
        return accuracy

    def load_model(self, name):
        if name in self._models:
            print(
                f"[ModelFactory.load_model] - model named {name} already exist. Returning it."
            )
            return self._models[name]
        # create path
        path = os.path.join("./models", f"{name}.tf")
        assert os.path.exists(
            path
        ), f"[ModelFactory.load_model] - can't load model {name}. Path {path} doesn't exist."
        model = load_model(
            path, custom_objects={"SentimentAnalyzer": SentimentAnalyzer}
        )
        with open(f"{path}/custom_config.json", "r") as f:
            custom_config = json.load(f)
        for (
            key,
            value,
        ) in (
            custom_config.items()
        ):  # a bit of a work-around, but this is the simplest way to have everything ok
            setattr(model, key, value)
        # add model
        self._models[name] = model

        return model

    def get_instantiated_model(self, name: str) -> Union["tf.keras.models.Model", None]:
        if not name in self._models:
            print(
                f"[ModelFactory.load_model] - model named {name} doesn't exist. Returning None.",
                file=sys.stderr,
            )
            return None
        return self._models[name]

    def predict_input_with_specific_model(self, model_name: str, text: str) -> str:
        if not model_name in self._models:
            # try to load the model
            self._models[model_name] = self.load_model(model_name)

        # if model still not available return empty sentiment
        if not model_name in self._models:
            print(
                f"[ModelFactory.predict_input_with_specific_model] - model named {model_name} doesn't exist. Returning empty sentiment.",
                file=sys.stderr,
            )
            return ""

        current_model = self._models[model_name]
        input_data_to_predict = prepare_input_sentence_to_predict(
            text,
            max_review_significant_length=current_model.max_review_significant_length,
        )

        prediction = current_model.predict(input_data_to_predict)
        prediction_label = np.argmax(prediction, axis=1).item()
        sentiment = LABEL_TRANSLATION_DICT[prediction_label]

        return sentiment
