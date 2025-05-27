# From: https://scholarly.readthedocs.io/en/stable/quickstart.html

# %%
from pprint import pprint

from scholarly import ProxyGenerator, scholarly


# # %%
def search_by_author(author):
    # Retrieve the author's data, fill-in, and print
    # Get an iterator for the author results
    search_query = scholarly.search_author(author)
    # Retrieve the first result from the iterator
    first_author_result = next(search_query)
    scholarly.pprint(first_author_result)

    # Retrieve all the details for the author
    author = scholarly.fill(first_author_result)
    scholarly.pprint(author)

    # Take a closer look at the first publication
    first_publication = author["publications"][0]
    first_publication_filled = scholarly.fill(first_publication)
    scholarly.pprint(first_publication_filled)

    # Print the titles of the author's publications
    publication_titles = [pub["bib"]["title"] for pub in author["publications"]]
    print(publication_titles)

    # Which papers cited that publication?
    citations = [citation["bib"]["title"] for citation in scholarly.citedby(first_publication_filled)]
    print(citations)

    return author, citations


if __name__ == "__main__":
    author = "Steven A Cholewiak"
    author, citations = search_by_author(author)
    pprint(author)
    pprint(citations)


# %%
def search_by_title(title):
    search_query = scholarly.search_pubs(title)
    # pub = next(search_query)
    # pprint(pub)
    pub_filled = scholarly.fill(next(search_query))
    # pprint(pub_filled)
    # pprint(pub_filled["num_citations"])
    citations = [citation for citation in scholarly.citedby(pub_filled)]
    # pprint(citations)
    return pub_filled, citations


if __name__ == "__main__":
    # title = "Emulating radiation transport on cosmological scale using a denoising Unet"
    title = "Neaural networks"
    pub_filled, citations = search_by_title(title)
    pprint(pub_filled)
    print()
    pprint(citations)
    print(len(citations))
    # print(citations[0])
    # print(citations[0]["bib"])
    # print(citations[0]["bib"].get("title"))
    # print(citations[0]["bib"].get("author"))
    # print(citations[0]["bib"].get("year"))
    # print(citations[0]["bib"].get("source"))
    # print(citations[0]["bib"].get("url"))
    # print(citations[0]["bib"].get("url_pdf"))
    # print(citations[0]["bib"].get("url_abs"))
    # print(citations[0]["bib"].get("url_html"))
    # print(citations[0]["bib"].get("url_pdf"))
    # print(citations[0]["bib"].get("url_abs"))
    # print(citations[0]["bib"].get("url_html"))

# %%


pg = ProxyGenerator()
scholarly.use_proxy(pg)

success = pg.SingleProxy(
    http="http://127.0.0.1:9090",
    https="http://127.0.0.1:9090",
)

search_query = scholarly.search_pubs(title)
pub = next(search_query)
pprint(pub)


# %%

pg = ProxyGenerator()
success = pg.Tor_Internal(tor_cmd="tor")
scholarly.use_proxy(pg)

# author = next(scholarly.search_author('Steven A Cholewiak'))
pub = next(scholarly.search_pubs(title))
pprint(pub)
