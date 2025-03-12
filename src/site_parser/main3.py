# %%
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from bs4 import BeautifulSoup
import time

# Настройки для Selenium
chrome_options = Options()
# chrome_options.add_argument("--headless")  # Запуск в фоновом режиме
chrome_options.add_argument("--disable-gpu")
chrome_options.add_argument("--no-sandbox")

# Инициализация драйвера
driver = webdriver.Chrome(options=chrome_options)

options = webdriver.ChromeOptions()
options.add_experimental_option(
    "prefs",
    {
        "download.default_directory": "~/Documents",  # Change default directory for downloads
        "download.prompt_for_download": False,  # To auto download the file
        "download.directory_upgrade": True,
        "plugins.always_open_pdf_externally": True,  # It will not show PDF directly in chrome
    },
)
# driver = webdriver.Chrome(options=options)


# %%
# URL страницы с вакансиями
url = "https://openreview.net/"
# url = 'https://hh.ru/search/vacancy?from=suggest_post&enable_snippets=false&area=113&education=not_required_or_not_specified&education=special_secondary&search_field=name&search_field=company_name&search_field=description&text=Excel аналитик&only_with_salary=true&page=1'
# Открываем страницу
driver.get(url)
# Ждем полной загрузки страницы
# Можно заменить на более умное ожидание, например, ожидание появления определенного элемента
time.sleep(5)
# Получаем HTML-код страницы
html = driver.page_source
# Закрываем браузер


# %%
# with open("index3.html", "w") as file:
#     file.write(html)

# %% Парсим HTML-код страницы
soup = BeautifulSoup(html, "html.parser")
# data = soup.find_all('div', class_='vacancy-info--ieHKDTkezpEj0Gsx')

allvenues = soup.find("section", attrs={"id": "all-venues"})
print(allvenues.name)
print(allvenues.attrs)
print()

result = soup.find_all("a", class_=True, href=True, recursive=True)

for tag in result:
    print(tag["href"], tag.text)

# %%
# elements = driver.find_element(By.TAG_NAME, 'section')
elements = driver.find_element(By.XPATH, "//section[@id='all-venues']")
# print(elements)

result = elements.find_elements(By.TAG_NAME, "a")
print(result)


for i, tag in enumerate(result):
    print(i, tag.text, tag.get_attribute("href"))


# %%
elements = driver.find_element(By.XPATH, "//section[@id='all-venues']")
print(elements)

# result = elements.find_elements(By.TAG_NAME, "a")
# print(result)


# for tag in result:
#     print(tag.text, tag.get_attribute("href"))


# %%
driver.get(url)
time.sleep(1)
html = driver.page_source
soup = BeautifulSoup(html, "html.parser")
groups = driver.find_element(By.XPATH, "//section[@id='all-venues']")
tags = groups.find_elements(By.TAG_NAME, "a")
print(tags)


for i, tag in enumerate(tags):
    if tag.text != "CARLA":
        continue
    print("=" * 100)
    print(tag.text, tag.get_attribute("href"))
    group_url = tag.get_attribute("href")

    print("=" * 100)
    driver.get(group_url)
    time.sleep(1)
    group_html = driver.page_source
    group_soup = BeautifulSoup(html, "html.parser")
    # print(group_soup)
    group_element = driver.find_element(By.XPATH, "//div[@id='group-container']")
    group_result = group_element.find_elements(By.TAG_NAME, "a")
    for tag in group_result:
        print(tag.text, tag.get_attribute("href"))

    print("=" * 100)
    driver.get(tag.get_attribute("href"))
    time.sleep(1)
    group_html = driver.page_source
    group_soup = BeautifulSoup(html, "html.parser")
    group_element = driver.find_element(By.XPATH, "//div[@id='group-container']")
    group_result = group_element.find_elements(By.TAG_NAME, "a")
    for tag in group_result:
        print(tag.text, tag.get_attribute("href"))

    print("=" * 100)
    driver.get(tag.get_attribute("href"))
    time.sleep(1)
    group_html = driver.page_source
    group_soup = BeautifulSoup(html, "html.parser")
    pdf_tags = driver.find_elements(By.XPATH, "//a[@title='Download PDF']")
    # pdf_tags = driver.find_elements(By.XPATH, "//a[@class='pdf link']")
    for pdf_tag in pdf_tags:
        pdf_link = pdf_tag.get_attribute("href")
        print(pdf_link)
        import urllib.request

        response = urllib.request.urlopen(pdf_link)
        file = open(f"{pdf_link.split('?')[-1]}.pdf", "wb")
        file.write(response.read())
        file.close()

    # group_element = driver.find_element(By.XPATH, "//div[@title='Download PDF']")
    # group_result = group_element.find_elements(By.TAG_NAME, "a")
    # for tag in group_result:
    #     print(tag.text, tag.get_attribute("href"))

    break


# %%
# driver.quit()
