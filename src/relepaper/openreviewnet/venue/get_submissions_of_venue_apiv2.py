# %%
import logging

logger = logging.getLogger(__name__)


def get_all_the_submissions_notes_of_venue_apiv2(client, venue_id):
    # invitation using get_all_notes().
    try:
        venue_group = client.get_group(venue_id)
        submission_name = venue_group.content["submission_name"]["value"]
        submissions = client.get_all_notes(
            invitation=f"{venue_id}/-/{submission_name}",
            details="replies",
        )
    except Exception:
        submissions = None
        logger.error("Error get all notes.")
    return submissions


def get_simple_all_the_submissions_notes_of_venue_apiv2(client, venue_id):
    # invitation using get_all_notes().
    try:
        submissions = client.get_all_notes(
            invitation=f"{venue_id}/-/Submission",
            details="replies",
        )
    except Exception:
        submissions = None
        logger.error("Error get all notes.")
    return submissions


def get_accepted_submissions_of_venue_apiv2(client, venue_id):
    # To only get "accepted" submissions, you'll need to query the notes by venueid.
    submissions = client.get_all_notes(
        content={"venueid": venue_id},
        details="replies",
    )
    return submissions


def get_active_submissions_under_review_of_venue_apiv2(client, venue_id):
    # To get active submissions under review:
    venue_group = client.get_group(venue_id)
    submission_name = venue_group.content["submission_name"]["value"]
    submissions = client.get_all_notes(
        invitation=f"{venue_id}/-/{submission_name}",
        details="replies",
    )
    return submissions


def get_withdrawn_submissions_of_venue_apiv2(client, venue_id):
    # To get withdrawn submissions.
    venue_group = client.get_group(venue_id)
    withdrawn_id = venue_group.content["withdrawn_venue_id"]["value"]
    submissions = client.get_all_notes(
        content={"venueid": withdrawn_id},
        details="replies",
    )
    return submissions


def get_desk_rejected_submissions_of_venue_apiv2(client, venue_id):
    # To get desk-rejected submissions.
    venue_group = client.get_group(venue_id)
    desk_rejected_venue_id = venue_group.content["desk_rejected_venue_id"]["value"]
    submissions = client.get_all_notes(
        content={"venueid": desk_rejected_venue_id},
        details="replies",
    )
    return submissions


# %%
if __name__ == "__main__":
    import sys

    logger.setLevel("DEBUG")
    formatter = logging.Formatter(
        "{asctime} :::: {levelname} :::: {name} :::: {module}:{funcName}:{lineno} \n>>> {message}",
        style="{",
    )
    stream_handler = logging.StreamHandler(stream=sys.stdout)
    stream_handler.setLevel("DEBUG")
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    # %%
    from relepaper.openreviewnet.clientapi.identify_client_api_version_for_venue import (
        identify_client_api_version_for_venue,
    )
    from relepaper.openreviewnet.clientapi.OpenReviewClients import OpenReviewClients

    clients = OpenReviewClients()
    venues = [
        # "NeurIPS.cc/2022/Conference",  # API1
        # "NeurIPS.cc/2024/Conference",  # API2
        # "conceptuccino.uni-osnabrueck.de/CARLA/2020/Workshop",  # API1
        # "ICLR.cc/2023/Workshop/Physics4ML",  # API1
        # "corl.org/2024/Workshop/MAPoDeL",  # API2
        # "ICLR.cc/2025/Workshop/SynthData",  # API2
        # "NeurIPS.cc/2020/Conference",  # None
        # "TMLR", # apiv2
        "thecvf.com/CVPR/2025/Workshop/GMCV",  # apiv2
    ]

    for i_venue, venue_id in enumerate(venues):
        api_version = identify_client_api_version_for_venue(clients, venue_id)

        logger.info(f"api_version: {api_version}, i_venus: {i_venue}, venue_id: {venue_id}")

        if api_version is None:
            continue

        client = clients.get_client(api_version=api_version)

        notes = []
        if api_version == "v1":
            pass
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

        for i_note, note in enumerate(notes[:20]):
            print(i_note, note.id, note.details.get("replies"))
