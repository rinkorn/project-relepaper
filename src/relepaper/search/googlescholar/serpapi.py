# %%
import json
from serpapi import GoogleScholarSearch

# search parameters
params = {
    "api_key": "Your SerpApi API key",
    "engine": "google_scholar_profiles",
    "hl": "en",  # language
    "mauthors": "biology",  # search query
}

search = GoogleScholarSearch(params)
results = search.get_dict()

# only first page results
for result in results["profiles"]:
    print(json.dumps(result, indent=2))

# # part of the output:
# '''
# {
#   "name": "Masatoshi Nei",
#   "link": "https://scholar.google.com/citations?hl=en&user=VxOmZDgAAAAJ",
#   "serpapi_link": "https://serpapi.com/search.json?author_id=VxOmZDgAAAAJ&engine=google_scholar_author&hl=en",
#   "author_id": "VxOmZDgAAAAJ",
#   "affiliations": "Laura Carnell Professor of Biology, Temple University",
#   "email": "Verified email at temple.edu",
#   "cited_by": 384074,
#   "interests": [
#     {
#       "title": "Evolution",
#       "serpapi_link": "https://serpapi.com/search.json?engine=google_scholar_profiles&hl=en&mauthors=label%3Aevolution",
#       "link": "https://scholar.google.com/citations?hl=en&view_op=search_authors&mauthors=label:evolution"
#     },
#     {
#       "title": "Evolutionary biology",
#       "serpapi_link": "https://serpapi.com/search.json?engine=google_scholar_profiles&hl=en&mauthors=label%3Aevolutionary_biology",
#       "link": "https://scholar.google.com/citations?hl=en&view_op=search_authors&mauthors=label:evolutionary_biology"
#     },
#     {
#       "title": "Molecular evolution",
#       "serpapi_link": "https://serpapi.com/search.json?engine=google_scholar_profiles&hl=en&mauthors=label%3Amolecular_evolution",
#       "link": "https://scholar.google.com/citations?hl=en&view_op=search_authors&mauthors=label:molecular_evolution"
#     },
#     {
#       "title": "Population genetics",
#       "serpapi_link": "https://serpapi.com/search.json?engine=google_scholar_profiles&hl=en&mauthors=label%3Apopulation_genetics",
#       "link": "https://scholar.google.com/citations?hl=en&view_op=search_authors&mauthors=label:population_genetics"
#     },
#     {
#       "title": "Phylogenetics",
#       "serpapi_link": "https://serpapi.com/search.json?engine=google_scholar_profiles&hl=en&mauthors=label%3Aphylogenetics",
#       "link": "https://scholar.google.com/citations?hl=en&view_op=search_authors&mauthors=label:phylogenetics"
#     }
#   ],
#   "thumbnail": "https://scholar.googleusercontent.com/citations?view_op=small_photo&user=VxOmZDgAAAAJ&citpid=3"
# }
# ... other results
# '''
# %%
