# %%
import logging

logger = logging.getLogger(__name__)


def get_all_venues_name(client):
    venues = client.get_group(id="venues").members
    return venues


# %%
if __name__ == "__main__":
    from openreviewstore.parsing.clientapi import OpenReviewClients, identify_client_api_version_for_venue

    clients = OpenReviewClients()
    venues = get_all_venues_name(clients.get_client("v2"))

    print("\n".join(venues[:10]))
