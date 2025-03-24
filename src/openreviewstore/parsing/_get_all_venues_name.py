# %%
import openreview


def get_all_venues_name(client):
    venues = client.get_group(id="venues").members
    return venues


def get_all_venue_notes(client, venue_id):
    # invitation using get_all_notes().
    venue_group = client.get_group(venue_id)
    submission_name = venue_group.content["submission_name"]["value"]
    submissions = client.get_all_notes(invitation=f"{venue_id}/-/{submission_name}")
    return submissions


def get_accepted_venue_notes(client, venue_id):
    # To only get "accepted" submissions, you'll need to query the notes by venueid.
    submissions = client.get_all_notes(content={"venueid": venue_id})
    return submissions


def get_active_under_review_venue_notes(client, venue_id):
    # To get active submissions under review:
    venue_group = client.get_group(venue_id)
    submission_name = venue_group.content["submission_name"]["value"]
    submissions = client.get_all_notes(invitation=f"{venue_id}/-/{submission_name}")
    return submissions


def get_withdrawn_venue_notes(client, venue_id):
    # To get withdrawn submissions:
    venue_group = client.get_group(venue_id)
    withdrawn_id = venue_group.content["withdrawn_venue_id"]["value"]
    submissions = client.get_all_notes(content={"venueid": withdrawn_id})
    return submissions


def get_desk_rejected_venue_notes(client, venue_id):
    # To get desk-rejected submissions:
    venue_group = client.get_group(venue_id)
    desk_rejected_venue_id = venue_group.content["desk_rejected_venue_id"]["value"]
    submissions = client.get_all_notes(content={"venueid": desk_rejected_venue_id})
    return submissions


def download_note_pdf(client, note):
    if note.content.get("pdf", {}).get("value"):
        f = client.get_attachment(note.id, "pdf")
        # file_name = f"pdfs/{note.id}-{note.number}.pdf"
        file_name = "pdfs/" + note.content.get("pdf", {}).get("value").split("/")[-1]
        with open(file_name, "wb") as op:
            op.write(f)


def download_note_supplementary_material(client, note):
    if note.content.get("supplementary_material", {}).get("value"):
        f = client.get_attachment(note.id, "supplementary_material")
        file_name = f"pdfs/{note.id}-{note.number}_supplementary_material.zip"
        with open(file_name, "wb") as op:
            op.write(f)


# %%
if __name__ == "__main__":
    client = openreview.Client(baseurl="https://api.openreview.net")

    venues = get_all_venues_name(client)
    # index = venues.index("corl.org/2024/Workshop/MAPoDeL")
    # venue_id = venues[index]
    venue_id = "ICLR.cc/2023/Workshop/Physics4ML"

    notes = get_accepted_venue_notes(client, venue_id)

    for note in notes:
        print(note.id, note.details)

    
    # for venue in venues:
    #     print(venue)

# %%
