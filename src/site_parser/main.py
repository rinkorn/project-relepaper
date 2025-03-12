# %%
import requests
from bs4 import BeautifulSoup
from fake_useragent import UserAgent

# %%
ua = UserAgent(
    browsers=["edge", "chrome", "firefox", "safari"],
    os=["Windows", "Linux", "Ubuntu", "Chrome OS", "Mac OS X"],
    min_percentage=1.3,
    # platforms=
    # min_version=
    # fallback=
)
print(ua.random)

# %%
openreview_url = "https://openreview.net/"
splash_url = lambda x: f"http://localhost:8050/render.html?url={x}"

url = splash_url(openreview_url)

# Определение заголовков, которые будут отправлены с запросом
# headers = {
#     "User-Agent": "Mozilla/5.0",  # Идентификация типа браузера, который отправляет запрос
#     "Accept": "text/html,application/xhtml+xml",  # Типы контента, которые клиент может обработать
#     # "Connection": "keep-alive",  # Указание на необходимость использования постоянного соединения
# }
# headers = {
#     "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
#     "Accept-Language": "ru-RU,ru;q=0.9,en-US;q=0.8,en;q=0.7",
#     "Accept-Encoding": "gzip, deflate, br",
# }
fake_ua = {
    "User-Agent": ua.random,
}
# Выполнение GET-запроса с установленными заголовками
response = requests.get(
    url=url,
    headers=fake_ua,
    timeout=10,
)
print(response.text)


# %%
with open("index.html", "wb") as file:
    file.write(response.content)

# %%
soup = BeautifulSoup(response.text, "html.parser")



# %%
allvenues = soup.find("section", attrs={"id": "all-venues"})
print(allvenues.name)
print(allvenues.attrs)
print()

result = soup.find_all("a", class_=True, href=True, recursive=True)

for tag in result:
    print(tag["href"], tag.text)

# %%
result = soup.select_one("section[id=all-venues]")
print(result)
print()
for item in result:
    print(item.text)

# %%
result = soup.find("section", attrs={"id": "all-venues"})
print(result)
print()
for tag in result:
    # print(tag["href"], item.text)
    print(tag)
