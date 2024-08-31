import requests

def update_status(id, images):
    url = f"http://localhost:8080/api/v1/generation-results/{id}/results"
    data = {
        "status": "COMPLETED",
        "images": images
    }
    print(f"::: Updating the generation job {id}")
    response = requests.post(url, json=data)
    print(f"::: Completed")
