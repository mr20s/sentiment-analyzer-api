from typing import List, Dict, Any, Union, Collection
import json
import tensorflow as tf
from keras.layers import (
    Embedding,
    Concatenate,
    LSTM,
    Dense,
    RepeatVector,
)
from keras.models import Model


class SentimentAnalyzer(Model):
    def __init__(
        self,
        max_review_significant_length: int,
        word_index_length: int,
        embedding_out_dims: int,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.max_review_significant_length = max_review_significant_length
        self.word_index_length = word_index_length
        self.embedding_out_dims = embedding_out_dims

        self.build(
            input_shape=(
                None,
                (
                    self.max_review_significant_length + 2
                ),  # 2 is the current number of numeric features (it may be parametric)
            )
        )

    def build(self, input_shape: Union["tf.TensorShape", List["tf.TensorShape"]]):
        self.repeat_numeric_input = RepeatVector(n=self.max_review_significant_length)
        self.embedding_layer = Embedding(
            input_dim=self.word_index_length,
            output_dim=self.embedding_out_dims,
            input_length=1,  # 1 review
            name=f"words_embedding",
        )
        self.concatenate_feats = Concatenate(name="concatenate_feats")
        self.lstm_1 = LSTM(80, name="lstm_1")
        self.output_classification_layer = Dense(
            3, activation="softmax", name="output_class_layer"
        )
        self.built = True

    def call(
        self, inputs: Union["tf.Tensor", Collection["tf.Tensor"]]
    ) -> Union["tf.Tensor", Collection["tf.Tensor"]]:
        embedding_input, numeric_input = inputs
        # treatment of tokenized words part of data
        x1 = self.embedding_layer(embedding_input)
        # treatment of numeric part of data
        x2 = self.repeat_numeric_input(numeric_input)
        # concatenating embeddings and numeric features
        concatenated = self.concatenate_feats([x1, x2])
        # LSTM-ing
        x = self.lstm_1(concatenated)
        # dense + softmax
        output = self.output_classification_layer(x)
        return output

    def get_custom_config(self) -> Dict[str, Any]:
        config = {
            "max_review_significant_length": self.max_review_significant_length,
            "word_index_length": self.word_index_length,
            "embedding_out_dims": self.embedding_out_dims,
        }
        return config

    def save(
        self,
        filepath,
        overwrite=True,
        include_optimizer=True,
        save_format=None,
        signatures=None,
        options=None,
        save_traces=True,
    ):
        if save_format is not None and save_format != "tf":
            raise ValueError(
                f"[SentimentAnalyzer.save] for this custom model, only 'tf' save format is supported"
            )
        super().save(
            filepath,
            overwrite,
            include_optimizer,
            save_format,
            signatures,
            options,
            save_traces,
        )
        custom_config = self.get_custom_config()
        with open(f"{filepath}/custom_config.json", "w") as f:
            json.dump(custom_config, f)
