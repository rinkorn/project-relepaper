# %%
import logging

from openreview import OpenReviewException

from relepaper.store.openreviewnet.clientapi.OpenReviewClients import OpenReviewClients

logger = logging.getLogger(__name__)


def identify_client_api_version_for_venue(clients: OpenReviewClients, venue_id: str) -> str:
    logger.info(f"Определение версии API для площадки: {venue_id}")

    api_version = None

    try:
        venue_group = clients.get_client("v2").get_group(venue_id)
        if venue_group.domain is not None:
            api_version = "v2"
            logger.info(f"Площадка {venue_id} доступна через API_v2 (присутствует поле domain)")
    except OpenReviewException as e:
        logger.error(f"Ошибка при проверке API_V2 для {venue_id}: {str(e)}")

    if api_version is not None:
        return api_version

    try:
        venue_group = clients.get_client("v2").get_group(venue_id)
        if venue_group.domain is None:
            venue_group = clients.get_client("v1").get_group(venue_id)
            api_version = "v1"
            logger.info(f"Площадка {venue_id} доступна через API_v1 (отсутствует поле domain)")
    except OpenReviewException as e:
        logger.error(f"Ошибка при проверке API_V1 для {venue_id}: {str(e)}")

    if api_version is None:
        logger.error(f"Ошибка при определении версию API для площадки {venue_id}")

    return api_version


# %%
if __name__ == "__main__":
    from relepaper.store.openreviewnet.note.simplify_note_content import simplify_note_content

    clients = OpenReviewClients()

    venues = [
        # "NeurIPS.cc/2022/Conference",  # API1
        # "NeurIPS.cc/2024/Conference",  # API2
        # "conceptuccino.uni-osnabrueck.de/CARLA/2020/Workshop",  # API1
        # "ICLR.cc/2023/Workshop/Physics4ML",  # API1
        # "corl.org/2024/Workshop/MAPoDeL",  # API2
        # "ICLR.cc/2025/Workshop/SynthData",  # API2
        # "NeurIPS.cc/2020/Conference",  # None
        "ISMIR.net/2018/WoRMS",
    ]

    for i_venue, venue_id in enumerate(venues):
        api_version = client_version_for_venue = identify_client_api_version_for_venue(clients, venue_id)
        if api_version is None:
            continue

        print(api_version, venue_id)
        client = clients.get_client(api_version)
        # venue_group = clients[client_version_for_venue].get_group(venue_id)
        submissions = client.get_all_notes(content={"venueid": venue_id})

        for i_note, note in enumerate(submissions):
            json_note = simplify_note_content(note.to_json())
            print(
                i_venue,
                i_note,
                json_note.get("number"),
                json_note.get("id"),
                json_note.get("content").get("pdf"),
                json_note.get("content").get("title"),
            )
