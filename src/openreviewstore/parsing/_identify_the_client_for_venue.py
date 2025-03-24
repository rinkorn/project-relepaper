# %%
import openreview

# %%
# must be frozen dict
clients = {
    "v1": openreview.Client(
        baseurl="https://api.openreview.net",
    ),
    "v2": openreview.api.OpenReviewClient(
        baseurl="https://api2.openreview.net",
    ),
}


# %%
def identify_the_client_for_venue(clients, venue_id):
    client_version_must_be = "v2"
    try:
        venue_group = clients["v2"].get_group(venue_id)
    except openreview.OpenReviewException as e:
        raise
    if venue_group.domain is None:
        client_version_must_be = "v1"
    return client_version_must_be


def simplify_note_content(note_dict):
    for k, v in note_dict["content"].items():
        if isinstance(v, dict) and v.get("value"):
            note_dict["content"][k] = v.get("value")
        else:
            note_dict["content"][k] = v
    return note_dict


# %%
venues = [
    # "NeurIPS.cc/2022/Conference",  # API1
    # "NeurIPS.cc/2024/Conference",  # API2
    "conceptuccino.uni-osnabrueck.de/CARLA/2020/Workshop",  # API1
    # "corl.org/2024/Workshop/MAPoDeL",  # API2
    # "ICLR.cc/2023/Workshop/Physics4ML",  # API1
    # "ICLR.cc/2025/Workshop/SynthData",  # API2
]

for i_venue, venue_id in enumerate(venues):
    version = client_version_for_venue = identify_the_client_for_venue(
        clients, venue_id
    )
    print(version, venue_id)

    # venue_group = clients[client_version_for_venue].get_group(venue_id)
    submissions = clients[version].get_all_notes(content={"venueid": venue_id})

    for i_note, note in enumerate(submissions):
        note_data = simplify_note_content(note.to_json())
        print(
            i_venue,
            i_note,
            note_data.get("number"),
            note_data.get("id"),
            note_data.get("content").get("pdf"),
            note_data.get("content").get("title"),
        )

