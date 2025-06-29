from dataclasses import dataclass

import requests

API_URL = "https://npucloud.tech"
HEADERS = {"Content-Type": "application/json"}


@dataclass
class DeleteModelResponse:
    status: bool
    err_message: str = ""


def delete_model(api_key: str, model_id: str):
    """
    Delete the model from NPUCloud.
    Parameters:
        - api_key: str
    Your API key. Get one at https://npucloud.tech/payments.php
        - model_id: str
    Your model_id. Check your models at https://npucloud.tech/models.php
    """
    payload = {"token": api_key, "model_id": model_id}
    resp = requests.post(f"{API_URL}/python_client/delete_model.php", json=payload, headers=HEADERS,
                         verify=True, timeout=15)
    if resp.status_code != 200:
        raise ValueError(f"Got status {resp.status_code} from the server while requesting the model deletion. "
                         f"Err msg: {resp.text}")
    try:
        resp = DeleteModelResponse(**resp.json())
    except ValueError as e:
        raise ValueError(f"Could not decode model delete_model task's response. Err msg: {resp.text}") from e
    if resp.status != "ok":
        raise ValueError(f"Could not delete the model {model_id}. Err message: {resp.err_message}")
