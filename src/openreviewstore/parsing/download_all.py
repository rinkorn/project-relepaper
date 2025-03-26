# %%
import argparse
import logging
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from openreviewstore.constants import PROJECT_PATH
from openreviewstore.parsing import (
    save_venue,
    save_note,
    OpenReviewClients,
    download_pdf_for_note,
    download_supplementary_material_for_note,
    get_accepted_submissions_for_double_blind_venues_apiv1,
    get_accepted_submissions_for_single_blind_venues_apiv1,
    get_accepted_submissions_of_venue_apiv2,
    get_active_submissions_for_a_double_blind_venue_apiv1,
    get_active_submissions_under_review_of_venue_apiv2,
    get_all_submissions_for_a_double_blind_venue_apiv1,
    get_all_the_submissions_notes_of_venue_apiv2,
    get_all_venues_name,
    get_desk_rejected_submissions_of_venue_apiv2,
    get_simple_all_the_submissions_notes_of_venue_apiv2,
    get_withdrawn_submissions_of_venue_apiv2,
    identify_client_api_version_for_venue,
    simplify_note_content,
)

logger = logging.getLogger(__name__)


def parse_args():
    pass


def process_note(note, venue_path, client, api_version):
    save_note(
        venue_path,
        note,
        additional_data={"details": note.details, "api_version": api_version},
        is_overwrite=False,
    )
    # save pdf
    download_pdf_for_note(
        note,
        venue_path,
        client,
        api_version,
        is_overwrite=False,
    )
    # # save supplementary
    download_supplementary_material_for_note(
        note,
        venue_path,
        client,
        api_version,
        is_overwrite=True,
    )


def main():
    store_path = PROJECT_PATH / "data/test_store/"

    clients = OpenReviewClients()

    all_venues = get_all_venues_name(clients.get_client("v2"))
    without_pdf_venues = []
    errored_venues = []

    # TODO: need create files with next information:
    # - обработанные мероприятия (мероприятие - количество notes - количество pdfs - есть ли прикрепленные данные)
    # - обработанные, но пустые мероприятия
    # - необработанные мероприятия (ошибки доступа по api)

    logger.info(f"\nlen(all_venues): {len(all_venues)}")

    testing_venues = [
        # "NeurIPS.cc/2020/Conference",  # None
        # "ICLR.cc/2013/conference",  # API1
        "ICLR.cc/2023/Workshop/Physics4ML",  # API1
        # "conceptuccino.uni-osnabrueck.de/CARLA/2020/Workshop",  # API1
        # "NeurIPS.cc/2022/Conference",  # API1
        # "NeurIPS.cc/2024/Conference",  # API2
        # "corl.org/2024/Workshop/MAPoDeL",  # API2
        # "ICLR.cc/2025/Workshop/SynthData",  # API2
        # "ISMIR.net/2018/WoRMS",  # API1
    ]

    # for i_venue, venue_id in enumerate(all_venues):
    # for i_venue, venue_id in enumerate(all_venues[700:2000:100]):
    for i_venue, venue_id in enumerate(testing_venues):
        time.sleep(1)

        api_version = identify_client_api_version_for_venue(clients, venue_id)

        if api_version is None:
            errored_venues.append(venue_id)
            continue

        client = clients.get_client(api_version=api_version)

        logger.info(f"api_version: {api_version}, i_venus: {i_venue}, venue_id: {venue_id}")

        venue_group = client.get_group(venue_id)
        venue_path = store_path / venue_id

        # save json_venue
        save_venue(
            venue_group,
            venue_path / "venue.json",
            additional_data={"api_version": api_version},
            is_overwrite=True,
        )

        notes = []
        if api_version == "v1":
            if not notes:
                notes = get_active_submissions_for_a_double_blind_venue_apiv1(client, venue_id)
            if not notes:
                notes = get_all_submissions_for_a_double_blind_venue_apiv1(client, venue_id)
            if not notes:
                notes = get_accepted_submissions_for_double_blind_venues_apiv1(client, venue_id)
            if not notes:
                notes = get_accepted_submissions_for_single_blind_venues_apiv1(client, venue_id)
        elif api_version == "v2":
            if not notes:
                notes = get_all_the_submissions_notes_of_venue_apiv2(client, venue_id)
            if not notes:
                notes = get_simple_all_the_submissions_notes_of_venue_apiv2(client, venue_id)
            if not notes:
                notes = get_accepted_submissions_of_venue_apiv2(client, venue_id)
            # if not notes:
            #     notes = get_active_submissions_under_review_of_venue_apiv2(client, venue_id)
            # if not notes:
            #     notes = get_withdrawn_submissions_of_venue_apiv2(client, venue_id)
            # if not notes:
            #     notes = get_desk_rejected_submissions_of_venue_apiv2(client, venue_id)

        if notes is None:
            errored_venues.append(venue_id)

        if not notes:
            continue

        # Параллельная обработка заметок с использованием ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=5) as executor:
            # for note in notes:
            for i_note, note in enumerate(notes):
                logger.debug(
                    f"i_note: {i_note}, note_number: {note.number}, note_id: {note.id}, note_quantites: {len(notes)}"
                )
                logger.debug(f"pdf: {note.content.get("pdf")}")
                logger.debug(f"supplementary_material: {note.content.get("supplementary_material")}")
                logger.debug(f"replies: {note.details.get("replies")}")
                executor.submit(process_note, note, venue_path, client, api_version)


if __name__ == "__main__":
    main()
