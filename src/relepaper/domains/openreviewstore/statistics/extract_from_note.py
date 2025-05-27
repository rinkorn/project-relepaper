# %%
import json
import logging
from datetime import datetime
from pathlib import Path
from pprint import pprint

logger = logging.getLogger(__name__)


# %%
def convert_timestamp_to_datetime(timestamp: int) -> datetime:
    return datetime.fromtimestamp(timestamp / 1000)


# %%
def extract_pdf_name_from_note_json(note_json_file: Path) -> str | None:
    """Извлечение pdf названия из <note>.json
    Args:
        note_json_file (Path): Путь к файлу

    Returns:
        str | None: Название pdf файла или None
    """
    with note_json_file.open("r", encoding="utf-8") as f:
        data = json.load(f)
        if "content" not in data:
            return None
        if "pdf" in data["content"] and isinstance(data["content"]["pdf"], str):
            return data["content"]["pdf"]
        if "pdf" in data["content"] and "value" in data["content"]["pdf"]:
            return data["content"]["pdf"]["value"]
        return None


def extract_replies_from_note_json(note_json_file: Path) -> list[dict] | None:
    """Извлечение replies из <note>.json. Replies - это все виды комментариев, отзывов, ответов и т.д.
    Args:
        note_json_file (Path): Путь к файлу

    Returns:
        list[dict] | None: Список replies или None
    """
    replies = []
    with note_json_file.open("r", encoding="utf-8") as f:
        data = json.load(f)
        if "details" not in data:
            return []
        if "replies" not in data["details"]:
            return []
        if not data["details"]["replies"]:
            return []
        replies.extend(data["details"]["replies"])
    return replies


def extract_responses_from_note_json(note_json_file: Path, invitation_endswith: str) -> list[dict]:
    """Извлечение responses из <note>.json. Responses - это определенные виды replies.
    Например, "Official_Review", "Author_Response", "Public_Comment", "Official_Comment", "Decision"

    Args:
        note_json_file (Path): Путь к файлу
        invitation_endswith (str): Конец строки в invitation.

    Returns:
        list[dict]: Список responses
    """
    responses = []
    replies = extract_replies_from_note_json(note_json_file)

    for reply in replies:
        invitation = reply.get("invitation", None)
        if invitation is None:
            continue
        if not isinstance(invitation, str):
            continue
        if invitation.endswith(invitation_endswith):
            responses.append(reply)

    for reply in replies:
        invitations = reply.get("invitations", None)
        if invitations is None:
            continue
        if isinstance(invitations, list):
            for invitation in invitations:
                if not isinstance(invitation, str):
                    continue
                if invitation.endswith(invitation_endswith):
                    responses.append(reply)
                    break

    return responses


def extract_dates_from_reply(reply: dict) -> list[datetime]:
    """Извлечение дат из replies
    Args:
        reply (dict): replies

    Returns:
        list[datetime]: список дат
    """
    # cdate - The Creation Date or cdate is a unix timestamp in milliseconds that can be set either in the past or in the future. It usually represents when the Edit was created.
    # tcdate - The True Creation Date or tcdate indicates the date in unix timestamp in milliseconds when the Edit is created. Unlike the cdate, its value cannot be set or modified by the user and it is not displayed in the UI.
    # mdate - The Modification Date or mdate shows when the Edit was last modified. The mdate value is a unix timestamp in milliseconds that can be set either in the past or in the future.
    # tmdate - The True Modification Date or tmdate indicates the date in unix timestamp in milliseconds when the Edit is modified. Unlike the mdate, its value cannot be set or modified by the user and it is not displayed in the UI.
    # ddate - The Deletion Date or ddate is used to soft delete an Edit. This means that Edits with a ddate value can be restored but will appear as deleted. The ddate value is a unix timestamp in milliseconds that can be set either in the past or in the future.

    dates = {}
    cdate = reply.get("cdate", None)
    tcdate = reply.get("tcdate", None)
    mdate = reply.get("mdate", None)
    tmdate = reply.get("tmdate", None)
    ddate = reply.get("ddate", None)
    dates["cdate"] = convert_timestamp_to_datetime(cdate) if cdate is not None else None
    dates["tcdate"] = convert_timestamp_to_datetime(tcdate) if tcdate is not None else None
    dates["mdate"] = convert_timestamp_to_datetime(mdate) if mdate is not None else None
    dates["tmdate"] = convert_timestamp_to_datetime(tmdate) if tmdate is not None else None
    dates["ddate"] = convert_timestamp_to_datetime(ddate) if ddate is not None else None
    return dates


if __name__ == "__main__":
    # note = Path("/data/data.sets/openreviewstore/ICLR.cc/2018/Conference/note/HkL7n1-0b.json")  # api1
    # note = Path("/data/data.sets/openreviewstore/ICLR.cc/2018/Conference/note/ryserbZR-.json")  # api1
    note = Path("/data/data.sets/openreviewstore/ICLR.cc/2024/Conference/note/0i6Z9N5MLY.json")  # api2
    pdf_name = extract_pdf_name_from_note_json(note)
    print("PDF Name: ", pdf_name)
    replies = extract_replies_from_note_json(note)
    print("Replies: ", len(replies), replies)
    official_reviews = extract_responses_from_note_json(note, "Official_Review")
    print("Official Reviews: ", len(official_reviews), official_reviews)
    meta_reviews = extract_responses_from_note_json(note, "Meta_Review")
    print("Meta Reviews: ", len(meta_reviews), meta_reviews)
    official_comments = extract_responses_from_note_json(note, "Official_Comment")
    print("Official Comments: ", len(official_comments), official_comments)
    author_responses = extract_responses_from_note_json(note, "Author_Response")
    print("Author Responses: ", len(author_responses), author_responses)
    public_comments = extract_responses_from_note_json(note, "Public_Comment")
    print("Public Comments: ", len(public_comments), public_comments)
    decisions = extract_responses_from_note_json(note, "Decision")
    print("Decisions: ", len(decisions), decisions)
    pprint(decisions[0])


if __name__ == "__main__":
    dates = extract_dates_from_reply(official_reviews[1])
    pprint(dates)
