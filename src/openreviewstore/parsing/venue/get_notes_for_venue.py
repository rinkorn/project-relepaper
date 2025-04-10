import logging
from typing import Any, List, Optional

from .get_submissions_of_venue_apiv1 import (
    get_accepted_submissions_for_double_blind_venues_apiv1,
    get_accepted_submissions_for_single_blind_venues_apiv1,
    get_active_submissions_for_a_double_blind_venue_apiv1,
    get_all_submissions_for_a_double_blind_venue_apiv1,
)
from .get_submissions_of_venue_apiv2 import (
    get_accepted_submissions_of_venue_apiv2,
    get_active_submissions_under_review_of_venue_apiv2,
    get_all_the_submissions_notes_of_venue_apiv2,
    get_simple_all_the_submissions_notes_of_venue_apiv2,
)

logger = logging.getLogger(__name__)


def get_notes_for_venue(client: Any, venue_id: str, api_version: str) -> Optional[List[Any]]:
    """Get a list of notes for a given venue, trying different API methods depending on the version.

    Args:
        client: OpenReview client
        venue_id: Venue ID
        api_version: API version

    Returns:
        Optional[List[Any]]: List of notes or None in case of an error
    """
    retrieval_methods = {
        "v1": [
            lambda: get_active_submissions_for_a_double_blind_venue_apiv1(client, venue_id),
            lambda: get_all_submissions_for_a_double_blind_venue_apiv1(client, venue_id),
            lambda: get_accepted_submissions_for_double_blind_venues_apiv1(client, venue_id),
            lambda: get_accepted_submissions_for_single_blind_venues_apiv1(client, venue_id),
        ],
        "v2": [
            lambda: get_all_the_submissions_notes_of_venue_apiv2(client, venue_id),
            lambda: get_simple_all_the_submissions_notes_of_venue_apiv2(client, venue_id),
            lambda: get_accepted_submissions_of_venue_apiv2(client, venue_id),
            lambda: get_active_submissions_under_review_of_venue_apiv2(client, venue_id),
            # lambda: get_withdrawn_submissions_of_venue_apiv2(client, venue_id),
            # lambda: get_desk_rejected_submissions_of_venue_apiv2(client, venue_id),
        ],
    }

    if api_version not in retrieval_methods:
        logger.error(f"Unsupported API version: {api_version} for venue {venue_id}", exc_info=True)
        return None

    notes = []

    for method in retrieval_methods[api_version]:
        try:
            result = method()
            if result:
                notes = result
                break
        except Exception as e:
            logger.error(f"Error getting notes for venue {venue_id}: {str(e)}", exc_info=True)

    return notes
