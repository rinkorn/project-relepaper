# %%
import logging

logger = logging.getLogger(__name__)


def get_active_submissions_for_a_double_blind_venue_apiv1(client, venue_id):
    # To get all 'active' submissions for a double-blind venue , pass your venue's blind submission
    # invitation into get_all_notes.
    submissions = client.get_all_notes(
        invitation=f"{venue_id}/-/Blind_Submission",
        details="replies",
        # details="directReplies",
    )
    return submissions


def get_all_submissions_for_a_double_blind_venue_apiv1(client, venue_id):
    # To get all submissions for a double-blind venue regardless of their status (active, withdrawn or
    # desk rejected), pass your venue's submission invitation to get_all_notes().
    submissions = client.get_all_notes(
        invitation=f"{venue_id}/-/Submission",
        details="replies",
        # details="directReplies",
    )
    return submissions


def get_accepted_submissions_for_double_blind_venues_apiv1(client, venue_id):
    # As a program organizer, to get only the "accepted" submissions for double-blind venues, query using
    # the Blind submission invitation and include 'directReplies' and 'original' in the details.

    # Double-blind venues

    submissions = client.get_all_notes(
        invitation=f"{venue_id}/-/Blind_Submission",
        details="replies,original",
        # details="directReplies,original",
    )
    blind_notes = {note.id: note for note in submissions}
    all_decision_notes = []
    for submission_id, submission in blind_notes.items():
        all_decision_notes = all_decision_notes + [
            reply
            for reply in submission.details["replies"]
            if reply["invitation"].endswith("Decision")
            # reply for reply in submission.details["directReplies"] if reply["invitation"].endswith("Decision")
        ]
    accepted_submissions = []
    for decision_note in all_decision_notes:
        if "Accept" in decision_note["content"]["decision"]:
            accepted_submissions.append(blind_notes[decision_note["forum"]].details["original"])
    return accepted_submissions


def get_accepted_submissions_for_single_blind_venues_apiv1(client, venue_id):
    # As a program organizer, to get only the "accepted" submissions, query using the Submission
    # invitation and include 'directReplies' in the details.

    # Single-blind venues

    submissions = client.get_all_notes(
        invitation=f"{venue_id}/-/Submission",
        details="replies",
    )
    notes = {note.id: note for note in submissions}
    all_decision_notes = []
    for submission_id, submission in notes.items():
        all_decision_notes = all_decision_notes + [
            reply for reply in submission.details["replies"] if reply["invitation"].endswith("Decision")
        ]
    accepted_submissions = []
    for decision_note in all_decision_notes:
        if "Accept" in decision_note["content"]["decision"]:
            accepted_submissions.append(notes[decision_note["forum"]])
    return accepted_submissions


# %%
if __name__ == "__main__":
    import openreview

    from relepaper.store.openreviewnet.clientapi.identify_client_api_version_for_venue import (
        identify_client_api_version_for_venue,
    )
    from relepaper.store.openreviewnet.clientapi.OpenReviewClients import OpenReviewClients

    clients = OpenReviewClients()
    venues = [
        # "ICLR.cc/2013/conference",  # API1
        "NeurIPS.cc/2020/Conference",
        # "ICLR.cc/2023/Workshop/Physics4ML",  # api_v1
        # "conceptuccino.uni-osnabrueck.de/CARLA/2020/Workshop",  # api_v1 - empty
        # "corl.org/2024/Workshop/MAPoDeL",  # api_v2
        # "ICLR.cc/2025/Workshop/SynthData",  # api_v2
        # "NeurIPS.cc/2021/Conference",  # api_v1
    ]
    for venue_id in venues:
        api_version = identify_client_api_version_for_venue(clients, venue_id)
        print(api_version)
        if api_version is None:
            continue

        client = clients.get_client(api_version=api_version)
        venue_group = client.get_group(venue_id)

        # notes = []
        # if api_version == "v1":
        #     if not notes:
        #         notes = get_active_submissions_for_a_double_blind_venue_apiv1(client, venue_id)
        #     if not notes:
        #         notes = get_all_submissions_for_a_double_blind_venue_apiv1(client, venue_id)
        #     if not notes:
        #         notes = get_accepted_submissions_for_double_blind_venues_apiv1(client, venue_id)
        #     if not notes:
        #         notes = get_accepted_submissions_for_single_blind_venues_apiv1(client, venue_id)
        # else:
        #     notes = []

        notes = client.get_all_notes(
            invitation=f"{venue_id}/-/Submission",
            details="replies",
            # details="directReplies",
        )

        for i_note, note in enumerate(notes[:5]):
            print(i_note, note.id, len(note.details["replies"]), note.details)
            print(openreview.tools.is_accept_decision())
