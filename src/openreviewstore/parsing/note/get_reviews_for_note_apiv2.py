# %%
import logging

logger = logging.getLogger(__name__)


def get_reviews_for_note_apiv2(note):
    # review_name = venue_group.content["review_name"]["value"]
    reviews = []
    for reply in note.details.get("replies"):
        # reply_json = openreview.api.Note.from_json(reply)
        reviews.append(reply)
        
        # if f"{venue_id}/{submission_name}{note.number}/-/{review_name}" in reply["invitations"]:
        #     reviews.append(reply)

        # #get the ID for the first review
        # example_review = reviews[0]
        # review_id = example_review["id"]
        # #get the revisions
        # edits = client.get_note_edits(review_id)
    return reviews


# %%
if __name__ == "__main__":
    import openreview
    from openreviewstore.parsing.clientapi import OpenReviewClients, identify_client_api_version_for_venue

    # clients = OpenReviewClients()
    venues = [
        # "ICLR.cc/2023/Workshop/Physics4ML",  # api_v1
        # "conceptuccino.uni-osnabrueck.de/CARLA/2020/Workshop",  # api_v1 - empty
        # "corl.org/2024/Workshop/MAPoDeL",  # api_v2
        # "ICLR.cc/2025/Workshop/SynthData",  # api_v2
        "NeurIPS.cc/2024/Conference",  # api_v2
    ]
    for venue_id in venues:
        api_version = identify_client_api_version_for_venue(clients, venue_id)
        print(api_version)

        client = clients.get_client(api_version=api_version)

        venue_group = client.get_group(venue_id)
        json_venue = venue_group.to_json()
        json_venue["api_version"] = api_version

        notes = []
        if api_version == "v1":
            pass
        elif api_version == "v2":
            submission_name = venue_group.content["submission_name"]["value"]
            submissions = client.get_all_notes(
                invitation=f"{venue_id}/-/{submission_name}",
                details="replies",
                # details="directReplies",
            )
            notes = submissions

        for i_note, note in enumerate(notes):
            print(i_note, note.id, note.details)

            json_note = note.to_json()
            json_note["details"] = note.details
            json_note["api_version"] = api_version

            reviews = []
            for reply in note.details["replies"]:
                reply_from_json = openreview.api.Note.from_json(reply)
                reviews.append(reply)
                # review_name = json_venue.get("content").get("review_name").get("value")
                # if f"{venue_id}/{submission_name}{note.number}/-/{review_name}" in reply["invitations"]:
                #     reviews.append(reply)

            print(len(reviews), len(json_note["details"]["replies"]))
            for review in reviews:
                print(review)
                
            #     review_id = review["id"]
            #     edits = client.get_note_edits(review_id)


            break
        break
