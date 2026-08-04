import requests

API_URL = "http://127.0.0.1:8000"


def predict(data):
    try:
        response = requests.post(
            f"{API_URL}/predict",
            json=data,
            timeout=30
        )

        response.raise_for_status()
        return response.json()

    except Exception as e:
        return {"error": str(e)}


def predict_csv(file):

    files = {
        "file": (
            file.name,
            file.getvalue(),
            "text/csv"
        )
    }

    response = requests.post(
        f"{API_URL}/predict_csv",
        files=files,
        timeout=120
    )

    response.raise_for_status()

    return response.content


def health():

    try:

        response = requests.get(
            f"{API_URL}/health",
            timeout=5
        )

        return response.json()

    except:

        return None