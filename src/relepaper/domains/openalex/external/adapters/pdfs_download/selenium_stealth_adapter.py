# %%
# From: https://stackoverflow.com/questions/68289474/selenium-headless-how-to-bypass-cloudflare-detection-using-selenium

import logging
import shutil
import time
import uuid

from selenium import webdriver
from selenium_stealth import stealth

from relepaper.domains.openalex.entities.pdf import OpenAlexPDF, PDFDownloadStrategy
from relepaper.domains.openalex.external.interfaces import IAdapter

logger = logging.getLogger(__name__)


class SeleniumStealthPDFDownloadAdapter(IAdapter):
    def download(
        self,
        openalex_pdf: OpenAlexPDF,
        timeout: int = 60,
    ) -> None:
        """Function to load pdf with selenium stealth"""

        # create dirname
        openalex_pdf.dirname.mkdir(parents=True, exist_ok=True)

        # create temp dirname
        dirname_temp = openalex_pdf.dirname / ("." + uuid.uuid4().hex)
        dirname_temp.mkdir(parents=True, exist_ok=True)

        # create options
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
                "download.default_directory": str(dirname_temp),  # Change default directory for downloads
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
        driver.get(openalex_pdf.url)

        # wait for download pdf
        total_time_sleep = 0
        delta_time_sleep = 0.2
        fname = ""
        while fname.lower().endswith(".pdf") is False:
            time.sleep(delta_time_sleep)
            total_time_sleep += delta_time_sleep
            for f in dirname_temp.glob("*.[pP][dD][fF]"):
                fname = f.name
            if total_time_sleep > timeout:
                logger.error(f"Timeout waiting for pdf: {openalex_pdf.url}")
                break

        # close driver
        driver.quit()

        # check if nothing downloading
        if not fname:
            if dirname_temp.is_dir():
                shutil.rmtree(dirname_temp)
            return None

        # move file to path_out
        path_in = dirname_temp / fname
        path_out = openalex_pdf.dirname / (openalex_pdf.filename or fname)
        if path_out.is_file():
            path_out.unlink()
        shutil.move(path_in, path_out)

        # remove temp_dirname
        if dirname_temp.is_dir():
            shutil.rmtree(dirname_temp)

        openalex_pdf.filename = path_out.name
        openalex_pdf.strategy = PDFDownloadStrategy.SELENIUM_STEALTH


# %%
if __name__ == "__main__":
    from relepaper.config.dev_settings import get_dev_settings
    from relepaper.domains.openalex.entities.pdf import OpenAlexPDF

    openalex_pdf = OpenAlexPDF(
        # url="https://openreview.net/pdf?id=CsCtO2YFn9",  # strategy: requests
        # url="https://arxiv.org/pdf/1912.01603",  # strategy: requests
        # url="https://www.pnas.org/doi/pdf/10.1073/pnas.0902281106",  # strategy: selenium
        # url="http://www.mcponline.org/article/S153594762034295X/pdf",  # strategy: selenium
        # url="https://doi.org/10.48550/arxiv.1912.01603",  # strategy: not any
        url="https://doi.org/10.1177/15.9.535",  # strategy: not any
        dirname=get_dev_settings().project_path / "data" / "pdf",
        # filename="special_filename.pdf",
    )

    service = SeleniumStealthPDFDownloadAdapter()
    service.download(openalex_pdf, 10)
    print(openalex_pdf.__dict__)
    print(openalex_pdf.is_downloaded)
    print(openalex_pdf.is_file_exist)
