# %%
import argparse
import logging
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import List

from relepaper.constants import PROJECT_PATH
from relepaper.store.openreviewnet import (
    OpenReviewClients,
    get_all_venues_name,
    get_notes_for_venue,
    identify_client_api_version_for_venue,
    process_note,
    save_venue,
)

logger = logging.getLogger(__name__)


def download_all(
    venue_ids: List[str] = [],
    store_path: Path = PROJECT_PATH / "store/",
    is_save_notes: bool = True,
    is_save_pdfs: bool = True,
    is_save_supplementaries: bool = False,
    is_overwrite_venue: bool = False,
    is_overwrite_notes: bool = False,
    is_overwrite_pdfs: bool = False,
    is_overwrite_supplementaries: bool = False,
    max_workers: int = 5,
) -> None:
    """Main function to download data from OpenReview.

    Args:
        venue_ids: List of venue IDs to process
        store_path: Path to save data
        max_workers: Maximum number of parallel processors
        is_save_notes: Flag to save note files
        is_save_pdfs: Flag to save PDF files
        is_save_supplementaries: Flag to save supplementary materials
        is_overwrite_venue: Flag to overwrite venue files
        is_overwrite_notes: Flag to overwrite note files
        is_overwrite_pdfs: Flag to overwrite PDF files
        is_overwrite_supplementaries: Flag to overwrite supplementary materials
    """

    # Initialize clients
    clients = OpenReviewClients()

    # Get all venues if no venue_ids are provided
    if not venue_ids:
        logger.info("Getting list of all venues...")
        venue_ids = get_all_venues_name(clients.get_client("v2"))
        logger.info(f"Found {len(venue_ids)} venues")
    else:
        logger.info(f"Processing specified venues: {len(venue_ids)}")

    # Process each venue
    for i_venue, venue_id in enumerate(venue_ids):
        time.sleep(1)

        # Identify API version
        api_version = identify_client_api_version_for_venue(clients, venue_id)
        if api_version is None:
            logger.error(f"Unsupported API version for venue {venue_id}")
            continue

        # Get client
        client = clients.get_client(api_version=api_version)

        logger.info(f"api_version: {api_version}, i_venue: {i_venue}, venue_id: {venue_id}")

        # Get venue group
        venue_group = client.get_group(venue_id)

        # Create venue path
        venue_path = Path(store_path) / venue_id
        venue_path.mkdir(parents=True, exist_ok=True)

        # Save venue info
        save_venue(
            venue_group=venue_group,
            path=venue_path / "venue.json",
            additional_data={"api_version": api_version},
            is_overwrite=is_overwrite_venue,
        )

        # Get notes for venue
        notes = get_notes_for_venue(
            client=client,
            venue_id=venue_id,
            api_version=api_version,
        )
        if notes is None or not notes:
            logger.info(f"No notes found for venue {venue_id}")
            continue

        # Process notes in parallel
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # for note in notes:
            for i_note, note in enumerate(notes):
                logger.debug(
                    f"i_note: {i_note}, note_number: {note.number}, note_id: {note.id}, note_quantites: {len(notes)}"
                )
                logger.debug(f"pdf: {note.content.get('pdf')}")
                logger.debug(f"supplementary_material: {note.content.get('supplementary_material')}")
                logger.debug(f"replies: {note.details.get('replies')}")
                executor.submit(
                    process_note,
                    note,
                    venue_path,
                    client,
                    api_version,
                    is_save_notes=is_save_notes,
                    is_save_pdfs=is_save_pdfs,
                    is_save_supplementaries=is_save_supplementaries,
                    is_overwrite_notes=is_overwrite_notes,
                    is_overwrite_pdfs=is_overwrite_pdfs,
                    is_overwrite_supplementaries=is_overwrite_supplementaries,
                )


def cli():
    parser = argparse.ArgumentParser(description="Скрипт для скачивания данных с OpenReview")

    parser.add_argument(
        "--store-path",
        type=str,
        default=str(PROJECT_PATH / "data/store/"),
        help="Путь для сохранения данных (по умолчанию: data/store/)",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=2,
        help="Максимальное количество параллельных обработчиков (по умолчанию: 5)",
    )
    parser.add_argument(
        "--venue-ids",
        type=str,
        nargs="+",
        default=None,
        help="ID мероприятий для обработки (можно указать несколько через пробел)",
    )
    parser.add_argument(
        "--save-notes",
        type=bool,
        default=True,
        help="Сохранять заметки (по умолчанию: True)",
    )
    parser.add_argument(
        "--save-pdfs",
        type=bool,
        default=True,
        help="Сохранять PDF-файлы (по умолчанию: True)",
    )
    parser.add_argument(
        "--save-supplementaries",
        type=bool,
        default=False,
        help="Сохранять дополнительные материалы (по умолчанию: False)",
    )
    parser.add_argument(
        "--overwrite-notes",
        type=bool,
        default=False,
        help="Перезаписать существующие файлы заметок",
    )
    parser.add_argument(
        "--overwrite-pdfs",
        type=bool,
        default=False,
        help="Перезаписать существующие PDF-файлы",
    )
    parser.add_argument(
        "--overwrite-supplementary",
        type=bool,
        default=False,
        help="Перезаписать существующие дополнительные материалы",
    )
    parser.add_argument(
        "--overwrite-venue",
        type=bool,
        default=False,
        help="Перезаписать существующие файлы информации о мероприятии",
    )
    parser.add_argument(
        "--overwrite-all",
        type=bool,
        default=False,
        help="Перезаписать все существующие файлы",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        help="Уровень логирования (INFO, DEBUG, WARNING, ERROR, CRITICAL)",
    )
    args = parser.parse_args()

    logging.basicConfig(level=args.log_level)

    download_all(
        venue_ids=args.venue_ids,
        store_path=args.store_path,
        is_save_notes=args.save_notes,
        is_save_pdfs=args.save_pdfs,
        is_save_supplementaries=args.save_supplementaries,
        is_overwrite_notes=args.overwrite_notes,
        is_overwrite_pdfs=args.overwrite_pdfs,
        is_overwrite_supplementaries=args.overwrite_supplementary,
        is_overwrite_venue=args.overwrite_venue,
        max_workers=args.max_workers,
    )


if __name__ == "__main__":
    cli()


# if __name__ == "__main__":
#     testing_args = argparse.Namespace(
#         venue_ids=[
#             # "NeurIPS.cc/2020/Conference",  # None
#             # "ICLR.cc/2013/conference",  # API1
#             "ICLR.cc/2023/Workshop/Physics4ML",  # API1
#             # "conceptuccino.uni-osnabrueck.de/CARLA/2020/Workshop",  # API1
#             # "NeurIPS.cc/2022/Conference",  # API1
#             # "NeurIPS.cc/2024/Conference",  # API2
#             # "corl.org/2024/Workshop/MAPoDeL",  # API2
#             # "ICLR.cc/2025/Workshop/SynthData",  # API2
#             # "ISMIR.net/2018/WoRMS",  # API1
#         ],
#         store_path=PROJECT_PATH / "data/test2_store/",
#         is_save_notes=True,
#         is_save_pdfs=True,
#         is_save_supplementaries=False,
#         is_overwrite_venue=True,
#         is_overwrite_notes=False,
#         is_overwrite_pdfs=False,
#         is_overwrite_supplementaries=False,
#         max_workers=5,
#     )
#     download_all(**vars(testing_args))
