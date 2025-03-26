# %%
import logging

logger = logging.getLogger(__name__)


def get_all_reviews_for_a_double_blind_venue_apiv1(venue_id):
    submissions = client.get_all_notes(
        invitation=f"{venue_id}/-/Blind_Submission",
        details="replies",
    )
    reviews = []
    for submission in submissions:
        reviews = reviews + [
            openreview.Note.from_json(reply)
            for reply in submission.details["replies"]
            if reply["invitation"].endswith("Official_Review")
        ]
    return reviews


def get_all_reviews_for_a_single_blind_venue_apiv1(venue_id):
    submissions = client.get_all_notes(
        invitation=f"{venue_id}/-/Submission",
        details="replies",
    )
    reviews = []
    for submission in submissions:
        reviews = reviews + [
            openreview.Note.from_json(reply)
            for reply in submission.details["replies"]
            if reply["invitation"].endswith("Official_Review")
        ]
    return reviews


# %%
if __name__ == "__main__":
    import openreview
    from openreviewstore.parsing.clientapi import (
        OpenReviewClients,
        identify_client_api_version_for_venue,
    )

    # clients = OpenReviewClients()
    venues = [
        # "ICLR.cc/2023/Workshop/Physics4ML",  # api_v1
        # "conceptuccino.uni-osnabrueck.de/CARLA/2020/Workshop",  # api_v1 - empty
        # "corl.org/2024/Workshop/MAPoDeL",  # api_v2
        # "ICLR.cc/2025/Workshop/SynthData",  # api_v2
        "NeurIPS.cc/2022/Conference",  # api_v1
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
            # submission_name = venue_group.content["submission_name"]["value"]
            submission_name = "Blind_Submission"
            # details_replies_key="directReplies"
            details_replies_key = "replies"
            submissions = client.get_all_notes(
                invitation=f"{venue_id}/-/{submission_name}",
                details=details_replies_key,
                # details="replies",
                # details="directReplies",
            )
            notes = submissions
        elif api_version == "v2":
            pass

        for i_note, note in enumerate(notes):
            print(i_note, note.id, note.details)

            json_note = note.to_json()
            json_note["details"] = note.details
            json_note["api_version"] = api_version

            reviews = []
            for reply in note.details[details_replies_key]:
                reply_from_json = openreview.api.Note.from_json(reply)
                reviews.append(reply)

            print(len(reviews), len(json_note["details"][details_replies_key]))
            for review in reviews:
                print(review)

            break
        break
