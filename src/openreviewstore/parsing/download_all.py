# %%
import time
import argparse
import logging
import sys
from pathlib import Path

from concurrent.futures import ThreadPoolExecutor

from openreviewstore.parsing import (
    OpenReviewClients,
    # download_note_pdf,
    # download_note_supplementary_material,
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
logger.setLevel("DEBUG")
formatter = logging.Formatter("%(asctime)s %(levelname)8s %(name)s | %(message)s")
stream_handler = logging.StreamHandler(stream=sys.stdout)
stream_handler.setLevel("DEBUG")
stream_handler.setFormatter(formatter)
logger.addHandler(stream_handler)


def parse_args():
    pass


def save_venue_group(path, venue_group, additional_data: dict, is_overwrite=False):
    import json

    if path.is_file() and not is_overwrite:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    # venue_path = store_path / (venue_group.id.replace("-", "_").replace("/", "-") + ".json")
    venue_dict = venue_group.to_json()
    venue_dict.update(additional_data)
    with open(path, "w", encoding="utf-8") as file:
        json.dump(venue_dict, file, indent=2, ensure_ascii=False)
    logger.info(f"venue_downloaded: {path}")


def save_note(venue_path, note, additional_data: dict, is_overwrite=False):
    import json

    note_path = venue_path / "note" / (note.id + ".json")
    if note_path.is_file() and not is_overwrite:
        return
    note_path.parent.mkdir(parents=True, exist_ok=True)
    # venue_path = store_path / (venue_group.id.replace("-", "_").replace("/", "-") + ".json")
    note_dict = note.to_json()
    note_dict.update(additional_data)
    with open(note_path, "w", encoding="utf-8") as file:
        json.dump(note_dict, file, indent=2, ensure_ascii=False)
    logger.info(f"note_downloaded: {note_path}")


def download_pdf_for_note(venue_path, note, client, api_version, is_overwrite=False):
    pdf_available_path = None
    if api_version == "v1":
        pdf_available_path = note.content.get("pdf", "")
    elif api_version == "v2":
        pdf_available_path = note.content.get("pdf", {}).get("value", "")
    if not pdf_available_path:
        return
    pdf_path = venue_path / Path(pdf_available_path).relative_to("/")
    # нужно ли перезаписать существующий файл
    if pdf_path.is_file() and not is_overwrite:
        return
    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    pdf = client.get_pdf(id=note.id)
    with open(pdf_path, "wb") as file:
        file.write(pdf)
    logger.info(f"pdf_downloaded: {pdf_path}")


def download_supplementary_material_for_note(venue_path, note, client, api_version, is_overwrite=False):
    supmat_available_path = None
    if api_version == "v1":
        supmat_available_path = note.content.get("supplementary_material", "")
    elif api_version == "v2":
        supmat_available_path = note.content.get("supplementary_material", {}).get("value", "")
    if not supmat_available_path:
        return
    supmat_path = venue_path / Path(supmat_available_path).relative_to("/")
    # нужно ли перезаписать существующий файл
    if supmat_path.is_file() and not is_overwrite:
        return
    supmat_path.parent.mkdir(parents=True, exist_ok=True)
    pdf = client.get_attachment(id=note.id, field_name="pdf")
    with open(supmat_path, "wb") as file:
        file.write(pdf)
    logger.info(f"supmat_downloaded: {supmat_path}")


# for note in notes[:2]:
#     download_note_supplementary_material(venue_path, note, client, is_overwrite=True)

# %%
if __name__ == "__main__":
    from openreviewstore.constants import PROJECT_PATH

    store_path = PROJECT_PATH / "data/store/"

    clients = OpenReviewClients()

    all_venues = get_all_venues_name(clients.get_client("v2"))
    without_pdf_venues = []
    errored_venues = []

    logger.info(f"\nlen(all_venues): {len(all_venues)}")

    testing_venues = [
        "NeurIPS.cc/2020/Conference",  # None
        "ICLR.cc/2013/conference",  # API1
        "ICLR.cc/2023/Workshop/Physics4ML",  # API1
        "conceptuccino.uni-osnabrueck.de/CARLA/2020/Workshop",  # API1
        "NeurIPS.cc/2022/Conference",  # API1
        "corl.org/2024/Workshop/MAPoDeL",  # API2
        "ICLR.cc/2025/Workshop/SynthData",  # API2
        "NeurIPS.cc/2024/Conference",  # API2
        "ISMIR.net/2018/WoRMS",
    ]

    for i_venue, venue_id in enumerate(all_venues):
        # for i_venue, venue_id in enumerate(all_venues[700:2000:100]):
        # for i_venue, venue_id in enumerate(testing_venues):

        time.sleep(1)

        api_version = identify_client_api_version_for_venue(clients, venue_id)

        if api_version is None:
            errored_venues.append(venue_id)
            continue

        client = clients.get_client(api_version=api_version)

        logger.info(f"api_version: {api_version}, i_venus: {i_venue}, venue_id: {venue_id}")

        venue_group = client.get_group(venue_id)
        # json_venue = venue_group.to_json()
        # json_venue["api_version"] = api_version
        # json_venue["notes_quantity"] = api_version

        venue_path = store_path / venue_id
        # venue_path = store_path / venue_id.replace("-", "_").replace("/", "-")

        # save json_venue
        save_venue_group(
            venue_path / "venue.json",
            venue_group,
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

        def process_note(client, api_version, note, venue_path):
            save_note(
                venue_path,
                note,
                additional_data={"details": note.details, "api_version": api_version},
                is_overwrite=False,
            )
            # save pdf
            download_pdf_from_note(
                venue_path,
                note,
                client,
                is_overwrite=False,
            )
            # # save supplementary
            # download_note_supplementary_material(
            #     venue_path,
            #     note,
            #     client,
            #     is_overwrite=True,
            # )

        # Параллельная обработка заметок с использованием ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=5) as executor:
            # for note in notes:
            for i_note, note in enumerate(notes):
                # logger.info(f"i_note: {i_note}, note_number: {note.number}, note_id: {note.id}, note_quantites: {len(notes)}")
                # logger.info(f"pdf: {note.content.get("pdf")}")
                # logger.info(f"supplementary_material: {note.content.get("supplementary_material")}")
                # logger.info(f"replies: {note.details.get("replies")}")
                executor.submit(
                    process_note,
                    client,
                    api_version,
                    note,
                    venue_path,
                )
        # break
