import logging
import re

import requests
from fake_useragent import UserAgent

from relepaper.domains.openalex.entities.pdf import OpenAlexPDF, PDFDownloadStrategy
from relepaper.domains.openalex.external.interfaces import IAdapter

logger = logging.getLogger(__name__)


class RequestsPDFDownloadAdapter(IAdapter):
    def download(
        self,
        openalex_pdf: OpenAlexPDF,
        timeout: int = 60,
    ) -> None:
        """Function to perform GET request with User-Agent spoofing"""

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
            response = requests.get(url=openalex_pdf.url, headers=headers, timeout=timeout)
        except Exception as e:
            logger.error(f"Error requesting pdf: {e}")

        if response.status_code == 200:
            content_disposition = response.headers.get("content-disposition", None)
            if content_disposition is None:
                return
            fname = re.findall("filename=(.+)", content_disposition)[0]
            fname = fname.replace('"', "") if openalex_pdf.filename is None else openalex_pdf.filename
            openalex_pdf.dirname.mkdir(parents=True, exist_ok=True)
            with open(str(openalex_pdf.dirname / fname), "wb") as f:
                f.write(response.content)
            openalex_pdf.filename = fname
            openalex_pdf.strategy = PDFDownloadStrategy.REQUESTS
        else:
            logger.error(f"Error status code: {response.status_code}")


# %%
if __name__ == "__main__":
    from relepaper.config.dev_settings import get_dev_settings
    from relepaper.domains.openalex.entities.pdf import OpenAlexPDF

    openalex_pdf = OpenAlexPDF(
        # url="https://openreview.net/pdf?id=CsCtO2YFn9",  # strategy: requests
        url="https://arxiv.org/pdf/1912.01603",  # strategy: requests
        # url="https://www.pnas.org/doi/pdf/10.1073/pnas.0902281106",  # strategy: selenium
        # url="http://www.mcponline.org/article/S153594762034295X/pdf",  # strategy: selenium
        # url="https://doi.org/10.48550/arxiv.1912.01603",  # strategy: not any
        # url="https://doi.org/10.1177/15.9.535",  # strategy: not any
        dirname=get_dev_settings().project_path / "data" / "pdf",
        # filename="special_filename.pdf",
    )

    service = RequestsPDFDownloadAdapter()
    service.download(openalex_pdf)
    print(openalex_pdf.__dict__)
    print(openalex_pdf.is_downloaded)
    print(openalex_pdf.is_file_exist)
