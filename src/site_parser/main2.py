# %%
import requests
from bs4 import BeautifulSoup

# %%
with open("index2.html", "rb") as f:
    html = f.read()

print(html)

# %%
soup = BeautifulSoup(html, 'html.parser')
title_tag = soup.title
print(title_tag.text)  # Выведет: Мой сайт


# %%
result = soup.find("section", attrs={"id": "all-venues"})
print(result)
print()
for tag in result:
    # print(tag["href"], item.text)
    print(tag)

# %%
allvenues = soup.find("section", attrs={"id": "all-venues"})
print(allvenues.name)
print(allvenues.attrs)
print()

result = soup.find_all("a", class_=True, href=True, recursive=True)

for tag in result:
    print(tag["href"], tag.text)