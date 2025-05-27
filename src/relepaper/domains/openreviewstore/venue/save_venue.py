# %%
import json
import logging

logger = logging.getLogger(__name__)


def save_venue(venue_group, path, additional_data: dict, is_overwrite=False):
    """
    Save venue to json file

    Args:
        venue_group: venue group
        path: path to save venue
        additional_data: additional data
    """
    result = {
        "success": False,
        "existed": False,
        "overwritten": False,
        "path": None,
        "error": False,
    }

    try:
        if path.is_file() and not is_overwrite:
            result["existed"] = True
            return result
        result["overwritten"] = True

        path.parent.mkdir(parents=True, exist_ok=True)

        # venue_path = store_path / (venue_group.id + ".json")
        venue_dict = venue_group.to_json()
        venue_dict.update(additional_data)
        with open(path, "w", encoding="utf-8") as file:
            json.dump(venue_dict, file, indent=2, ensure_ascii=False)
        logger.info(f"Venue saved: {path}")
    except Exception as e:
        result["error"] = True
        logger.error(f"Error saving venue {venue_group.id}: {str(e)}")

    return result
