# %%
# From: https://stackoverflow.com/questions/68289474/selenium-headless-how-to-bypass-cloudflare-detection-using-selenium

import logging
import re
import time
from pathlib import Path

import requests
from fake_useragent import UserAgent
from selenium import webdriver
from selenium_stealth import stealth

logger = logging.getLogger(__name__)


def load_pdf_with_requests(url: str, path_out: Path = Path.cwd(), timeout: int = 10):
    """Function to perform GET request with User-Agent spoofing"""
    path_out.mkdir(parents=True, exist_ok=True)

    ua = UserAgent(
        browsers=["chrome", "firefox", "safari"],
        os=["Windows", "Linux", "Ubuntu", "Chrome OS", "Mac OS X", "Android", "iOS"],
        min_percentage=1.3,
        # platforms=
        # min_version=
        # fallback=
    )
    headers = {
        "Accept": "application/pdf",
        "User-Agent": ua.random,
    }
    try:
        response = requests.get(
            url=url,
            headers=headers,
            timeout=timeout,
        )
    except Exception as e:
        logger.error(f"Error requesting pdf: {e}")
        return

    if response.status_code == 200:
        fname = re.findall("filename=(.+)", response.headers["content-disposition"])[0]
        fname = fname.replace('"', "")
        with open(str(path_out / fname), "wb") as f:
            f.write(response.content)
    else:
        logger.error(f"Error status code: {response.status_code}")
        return
    return True


# %%
def load_pdf_with_selenium(url, path_out: Path = Path.cwd(), sleep_time: int = 10):
    """Function to load pdf with selenium"""
    path_out.mkdir(parents=True, exist_ok=True)

    options = webdriver.ChromeOptions()
    options.add_argument("--headless=new")
    options.add_argument(
        "user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36"
    )
    # options.add_argument(
    #     "user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/96.0.4664.110 Safari/537.36"
    # )
    options.add_experimental_option(
        "prefs",
        {
            "plugins.plugins_disabled": ["Chrome PDF Viewer"],
            "plugins.always_open_pdf_externally": True,  # It will not show PDF directly in chrome
            "download.default_directory": str(path_out),  # Change default directory for downloads
            "download.prompt_for_download": False,  # To auto download the file
            "download.directory_upgrade": True,
        },
    )
    # Start the WebDriver instance with the options
    driver = webdriver.Chrome(options=options)
    driver.get(url)
    time.sleep(sleep_time)
    driver.quit()
    return True


# %%
def load_pdf_with_selenium_stealth(url, path_out: Path = Path.cwd(), sleep_time: int = 10):
    """Function to load pdf with selenium stealth"""
    path_out.mkdir(parents=True, exist_ok=True)

    options = webdriver.ChromeOptions()
    options.add_argument("start-maximized")
    options.add_argument("--headless")
    options.add_experimental_option("excludeSwitches", ["enable-automation"])
    options.add_experimental_option("useAutomationExtension", False)
    options.add_experimental_option(
        "prefs",
        {
            "plugins.plugins_disabled": ["Chrome PDF Viewer"],
            "plugins.always_open_pdf_externally": True,  # It will not show PDF directly in chrome
            "download.default_directory": str(path_out),  # Change default directory for downloads
            "download.prompt_for_download": False,  # To auto download the file
            "download.directory_upgrade": True,
        },
    )
    driver = webdriver.Chrome(options=options)
    stealth(
        driver,
        languages=["en-US", "en"],
        vendor="Google Inc.",
        platform="Win32",
        webgl_vendor="Intel Inc.",
        renderer="Intel Iris OpenGL Engine",
        fix_hairline=True,
    )
    driver.get(url)
    time.sleep(sleep_time)
    driver.quit()
    return True


# %%
if __name__ == "__main__":
    from relepaper.config import PROJECT_PATH

    path_out = PROJECT_PATH / "data" / "pdf"

    # pdf_url = "https://openreview.net/pdf?id=CsCtO2YFn9"
    # load_pdf_with_requests(pdf_url, path_out=path_out, timeout=3)

    # pdf_url = "https://www.pnas.org/doi/pdf/10.1073/pnas.0902281106"
    pdf_url = "https://doi.org/10.1177/15.9.535"
    load_pdf_with_selenium(pdf_url, path_out, sleep_time=3)

    # pdf_url = "http://www.mcponline.org/article/S153594762034295X/pdf"
    # pdf_url = "https://doi.org/10.1177/15.9.535"
    # load_pdf_with_selenium_stealth(pdf_url, path_out, sleep_time=3)


# %%
