# From: https://medium.com/@darshankhandelwal12/scrape-google-scholar-using-python-3f35a3a6597b

# %%
import requests
from bs4 import BeautifulSoup

# url = "https://scholar.google.com/citations?user=VxOmZDgAAAAJ&hl=en"
title = "Emulating radiation transport on cosmological scale using a denoising Unet"


# %%
def getScholarData():
    try:
        url = "https://www.google.com/scholar?q=Quantum+Physics&hl=en"
        headers = {
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_11_3) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/100.0.4896.127 Safari/537.36"
        }
        response = requests.get(url, headers=headers)
        soup = BeautifulSoup(response.text, "html.parser")
        return soup
    except Exception as e:
        print(e)


if __name__ == "__main__":
    soup = getScholarData()
    print(soup)


# %%
def getScholarData():
    try:
        url = "https://www.google.com/scholar?q=Quantum+Physics&hl=en"
        headers = {
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_11_3) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/100.0.4896.127 Safari/537.36"
        }
        response = requests.get(url, headers=headers)
        soup = BeautifulSoup(response.text, "html.parser")

        scholar_results = []

        for el in soup.select(".gs_r"):
            scholar_results.append(
                {
                    "title": el.select(".gs_rt")[0].text,
                    "title_link": el.select(".gs_rt a")[0]["href"],
                    "id": el.select(".gs_rt a")[0]["id"],
                    "displayed_link": el.select(".gs_a")[0].text,
                    "snippet": el.select(".gs_rs")[0].text.replace("\n", ""),
                    "cited_by_count": el.select(".gs_nph+ a")[0].text,
                    "cited_link": "https://scholar.google.com" + el.select(".gs_nph+ a")[0]["href"],
                    "versions_count": el.select("a~ a+ .gs_nph")[0].text,
                    "versions_link": "https://scholar.google.com" + el.select("a~ a+ .gs_nph")[0]["href"]
                    if el.select("a~ a+ .gs_nph")[0].text
                    else "",
                }
            )

        for i in range(len(scholar_results)):
            scholar_results[i] = {
                key: value for key, value in scholar_results[i].items() if value != "" and value is not None
            }

        return scholar_results

    except Exception as e:
        print(e)


if __name__ == "__main__":
    scholar_results = getScholarData()
    print(scholar_results)


# %%
def getData():
    try:
        url = "https://scholar.google.com/scholar?q=info:cU32d0ZoSA0J:scholar.google.com&output=cite"
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/104.0.0.0 Safari/537.36"
        }
        response = requests.get(url, headers=headers)
        soup = BeautifulSoup(response.content, "html.parser")
        cite_results = []
        for el in soup.select("#gs_citt tr"):
            cite_results.append(
                {"title": el.select_one(".gs_cith").text.strip(), "snippet": el.select_one(".gs_citr").text.strip()}
            )
        links = []
        for el in soup.select("#gs_citi .gs_citi"):
            links.append({"name": el.text.strip(), "link": el.get("href")})
        return cite_results, links
    except Exception as e:
        print(e)


if __name__ == "__main__":
    cite_results, links = getData()

    for i in range(len(cite_results)):
        print(cite_results[i])

    for i in range(len(links)):
        print(links[i])


# %%
def getScholarProfiles():
    try:
        url = "https://scholar.google.com/citations?hl=en&view_op=search_authors&mauthors=Quantum+Physics"
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/101.0.4951.54 Safari/537.36"
        }
        response = requests.get(url, headers=headers)
        soup = BeautifulSoup(response.content, "html.parser")

        scholar_profiles = []
        for el in soup.select(".gsc_1usr"):
            profile = {
                "name": el.select_one(".gs_ai_name").get_text(),
                "name_link": "https://scholar.google.com" + el.select_one(".gs_ai_name a")["href"],
                "position": el.select_one(".gs_ai_aff").get_text(),
                "email": el.select_one(".gs_ai_eml").get_text(),
                "departments": el.select_one(".gs_ai_int").get_text(),
                "cited_by_count": el.select_one(".gs_ai_cby").get_text().split(" ")[2],
                # "title": el.select(".gs_rt")[0].text,
                # "title_link": el.select(".gs_rt a")[0]["href"],
                # "id": el.select(".gs_rt a")[0]["id"],
                # "displayed_link": el.select(".gs_a")[0].text,
                # "snippet": el.select(".gs_rs")[0].text.replace("\n", ""),
                # "cited_by_count": el.select(".gs_nph+ a")[0].text,
                # "cited_link": "https://scholar.google.com" + el.select(".gs_nph+ a")[0]["href"],
                # "versions_count": el.select("a~ a+ .gs_nph")[0].text,
                # "versions_link": "https://scholar.google.com" + el.select("a~ a+ .gs_nph")[0]["href"] if el.select("a~ a+ .gs_nph")[0].text else "",
            }
            scholar_profiles.append({k: v for k, v in profile.items() if v})
        return scholar_profiles
    except Exception as e:
        print(e)


if __name__ == "__main__":
    scholar_profiles = getScholarProfiles()
    for i in range(len(scholar_profiles)):
        print(scholar_profiles[i])


# %%
def getAuthorProfileData():
    try:
        url = "https://scholar.google.com/citations?hl=en&user=Pn8ouvAAAAAJ"
        headers = {
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_13_5) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/100.0.4896.127 Safari/537.36"
        }
        response = requests.get(url, headers=headers)
        print(response.status_code)
        soup = BeautifulSoup(response.text, "html.parser")
        author_results = {}
        author_results["name"] = soup.select_one("#gsc_prf_in").get_text()
        author_results["position"] = soup.select_one("#gsc_prf_inw+ .gsc_prf_il").text
        author_results["email"] = soup.select_one("#gsc_prf_ivh").text
        author_results["published_content"] = soup.select_one("#gsc_prf_int").text
        return author_results
    except Exception as e:
        print(e)


if __name__ == "__main__":
    author_results = getAuthorProfileData()
    print(author_results)


# %%
def getAuthorProfileData():
    try:
        url = "https://scholar.google.com/citations?hl=en&user=cOsxSDEAAAAJ"
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/104.0.0.0 Safari/537.36"
        }
        response = requests.get(url, headers=headers)
        soup = BeautifulSoup(response.text, "html.parser")
        author_results = {}
        articles = []
        author_results["name"] = soup.select_one("#gsc_prf_in").text
        author_results["position"] = soup.select_one("#gsc_prf_inw+ .gsc_prf_il").text
        author_results["email"] = soup.select_one("#gsc_prf_ivh").text
        author_results["departments"] = soup.select_one("#gsc_prf_int").text
        for el in soup.select("#gsc_a_b .gsc_a_t"):
            article = {
                "title": el.select_one(".gsc_a_at").text,
                "link": "https://scholar.google.com" + el.select_one(".gsc_a_at")["href"],
                "authors": el.select_one(".gsc_a_at+ .gs_gray").text,
                "publication": el.select_one(".gs_gray+ .gs_gray").text,
            }
            articles.append(article)
        for i in range(len(articles)):
            articles[i] = {k: v for k, v in articles[i].items() if v and v != ""}
        cited_by = {}
        cited_by["table"] = []
        cited_by["table"].append({})
        cited_by["table"][0]["citations"] = {}
        cited_by["table"][0]["citations"]["all"] = soup.select_one("tr:nth-child(1) .gsc_rsb_sc1+ .gsc_rsb_std").text
        cited_by["table"][0]["citations"]["since_2017"] = soup.select_one(
            "tr:nth-child(1) .gsc_rsb_std+ .gsc_rsb_std"
        ).text
        cited_by["table"].append({})
        cited_by["table"][1]["h_index"] = {}
        cited_by["table"][1]["h_index"]["all"] = soup.select_one("tr:nth-child(2) .gsc_rsb_sc1+ .gsc_rsb_std").text
        cited_by["table"][1]["h_index"]["since_2017"] = soup.select_one(
            "tr:nth-child(2) .gsc_rsb_std+ .gsc_rsb_std"
        ).text
        cited_by["table"].append({})
        cited_by["table"][2]["i_index"] = {}
        cited_by["table"][2]["i_index"]["all"] = soup.select_one("tr~ tr+ tr .gsc_rsb_sc1+ .gsc_rsb_std").text
        cited_by["table"][2]["i_index"]["since_2017"] = soup.select_one("tr~ tr+ tr .gsc_rsb_std+ .gsc_rsb_std").text
        return author_results, articles, cited_by

    except Exception as e:
        print(e)


if __name__ == "__main__":
    author_results, articles, cited_by = getAuthorProfileData()
    print(author_results)
    for i in range(len(articles)):
        print(articles[i])
    for i in range(len(cited_by["table"])):
        print(cited_by["table"][i])
