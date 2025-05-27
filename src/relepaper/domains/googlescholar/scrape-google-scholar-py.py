from google_scholar_py import CustomGoogleScholarProfiles
import json

# # %% Example usage SerpApi backend
# from google_scholar_py import SerpApiGoogleScholarOrganic
# import json

# profile_parser = SerpApiGoogleScholarProfiles()
# data = profile_parser.scrape_google_scholar_profile_results(
#     query="blizzard",
#     api_key="your-serpapi-api-key",  # https://serpapi.com/manage-api-key
#     pagination=False,
#     # other params
# )
# print(json.dumps(data, indent=2))


# %%
parser = CustomGoogleScholarProfiles()
data = parser.scrape_google_scholar_profiles(
    # query="blizzard",
    query="Emulating radiation transport on cosmological scale using a denoising Unet",
    pagination=False,
    save_to_csv=False,
    save_to_json=False,
)
print(json.dumps(data, indent=2))
