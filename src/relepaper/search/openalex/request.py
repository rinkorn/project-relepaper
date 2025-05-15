# %%
import requests

doi = "10.48550/arXiv.2205.01833"

url = f"https://api.openalex.org/works?filter=doi:{doi}"
r = requests.get(url)
response_data = r.json()
openalex_article = response_data["results"][0]

print(f"Within the OpenAlex data, the OpenAlex paper has {openalex_article['cited_by_count']} (incoming) citations.")

# %%
