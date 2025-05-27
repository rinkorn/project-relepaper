# %%
import requests
import json

url = "https://api.scrapeless.com/api/v1/scraper/request"

payload = json.dumps(
    {
        "actor": "scraper.google.scholar",
        "input": {
            "engine": "google_scholar",
            "q": "biology",
        },
    }
)
headers = {
    "Content-Type": "application/json",
}

response = requests.request(
    "POST",
    url,
    headers=headers,
    data=payload,
)

print(response.text)

# %%
