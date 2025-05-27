from pathlib import Path

from relepaper.domains.openalex.services.pdf_download_service_factory import PDFDownloadServiceFactory
from relepaper.domains.openalex.services.pdf_create_service import OpenAlexPDFCreateService
from relepaper.presenters.interfaces import IPresenter
from relepaper.views.interfaces import IView


class PDFDownloadPresenter(IPresenter):
    def __init__(
        self,
        view: IView,
        pdf_entity_service: OpenAlexPDFCreateService,
        download_service_factory: PDFDownloadServiceFactory,
    ):
        self._view = view
        self._pdf_entity_service = pdf_entity_service
        self._download_service_factory = download_service_factory

    def download_pdf(self, url: str, dirname: str, strategy: str):
        # 1. Создание сущности через сервис
        pdf_entity = self._pdf_entity_service.create_pdf_entity(url, Path(dirname))

        # 2. Получение сервиса загрузки
        download_service = self._download_service_factory.create(strategy)

        # 3. Выполнение загрузки
        download_service.download_pdf(pdf_entity)

        # 4. Отображение результата
        self._view.display_result(pdf_entity)
