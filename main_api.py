from fastapi import FastAPI, Body, HTTPException
from typing import Optional
import time
import uvicorn

from model_lib import ModelFactory


MODEL_FACTORY_INSTANCE = ModelFactory.get_instance()


app = FastAPI()

# NOTE: here other routes can be added to retrieve the list of existing models, list of available datasets, upload other datasets, and so on...


@app.get("/")
def index():
    return {"message": "Welcome to Reviews' Sentiment Analyzer"}


@app.post("/new_model")
async def new_model(
    embedding_out_dims: Optional[int] = Body(
        default=None,
        description="Number of embeddings' dimensions. Hyper-parameter used to create the new model. If None, it is the square root of vocabulary size",
    ),
    data_path: str = Body(
        default="./data/dataset.csv",
        description="Path where data to train/validate/test the new model is located.",
    ),
    name: str = Body(
        default="sentimentANN",
        description=(
            "Name to give to the new model. If a model with this name already exist, "
            "you will be prompted for a new name during the persistence of the model (save_model = True). "
            "However, if a model with the same name has been already loaded, the new one overwrites takes its place."
        ),
    ),
    save_model: bool = Body(
        default=True, description="Whether to save the created model or not."
    ),
    epochs: int = Body(
        default=10,
        description="Number of epochs to train the model. Early stopping patience will be 10% of this number.",
    ),
    batch_size: int = Body(
        default=128, description="Batch size employed during training."
    ),
    test_model: bool = Body(
        default=False,
        description="Specify if test the new model on test set generated from data in data_path, or not. If True, accuracy will be returned",
    ),
):
    result_dict = dict()
    # preparing data
    hyper_params = MODEL_FACTORY_INSTANCE.prepare_current_data(data_path=data_path)
    result_dict.update(hyper_params)  # add computed hyper params to the result
    # start training
    start_time = time.time()
    _ = MODEL_FACTORY_INSTANCE.create_and_train_new_model(
        max_review_significant_length=hyper_params["max_review_significant_length"],
        word_index_length=hyper_params["word_index_length"],
        embedding_out_dims=embedding_out_dims,
        name=name,
        save_model=save_model,
        epochs=epochs,
        batch_size=batch_size,
    )
    end_time = time.time()
    result_dict.update({"training_time_seconds": end_time - start_time})

    if test_model:
        test_accuracy = MODEL_FACTORY_INSTANCE.test_model_accuracy(name)
        result_dict.update({"accuracy_on_test_set": test_accuracy})

    return result_dict


@app.post("/predict")
async def predict(
    text: str = Body(..., description="analyze sentiment of this text"),
    model_name: str = Body(
        ..., description="Model name to use for predicitons. It must be available"
    ),
):
    if model_name is None:
        raise HTTPException(
            status_code=400, detail="The name of the model to employ is mandatory"
        )

    sentiment = MODEL_FACTORY_INSTANCE.predict_input_with_specific_model(
        model_name, text
    )

    return {"sentiment": sentiment}


@app.get("/get_available_models")
async def get_available_models():
    return {"model_names": MODEL_FACTORY_INSTANCE.get_loadable_model_names()}


# DEBUG
# import asyncio
# sentiment = asyncio.run(
#     predict(
#         text="Man nepatiko, kaip Sophia tvarkė mano nagus, o salonas nebuvo valymo viršūnė. Netgi netvarkingas.",
#         model_name="20sent_smaller"
#     ),
#     debug=True
# )
# print(sentiment)


if __name__ == "main":
    uvicorn.run(app, host="127.0.0.1", port=8000)
