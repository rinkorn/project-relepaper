# %%
import logging
from typing import Any, Dict

from relepaper.openreviewnet.note.download_pdf_for_note import download_pdf_for_note
from relepaper.openreviewnet.note.download_supplementaries_material_for_note import (
    download_supplementaries_material_for_note,
)
from relepaper.openreviewnet.note.save_note import save_note

logger = logging.getLogger(__name__)


def process_note(
    note,
    venue_path,
    client,
    api_version,
    is_save_notes: bool = True,
    is_save_pdfs: bool = True,
    is_save_supplementaries: bool = False,
    is_overwrite_notes: bool = False,
    is_overwrite_pdfs: bool = False,
    is_overwrite_supplementaries: bool = False,
) -> Dict[str, Any]:
    """Process a single note and download associated data.

    Args:
        note: OpenReview note object
        venue_path: Path to save data
        client: OpenReview client
        api_version: API version
        is_save_notes: Flag to save note files
        is_save_pdfs: Flag to save PDF files
        is_save_supplementaries: Flag to save supplementary materials
        is_overwrite_notes: Flag to overwrite note files
        is_overwrite_pdfs: Flag to overwrite PDF files
        is_overwrite_supplementaries: Flag to overwrite supplementary materials

    Returns:
        Dict[str, Any]: Dictionary with processing statistics
    """

    result = {
        "note_success": False,  # Note saved successfully
        "note_existed": False,  # Note already existed
        "note_overwritten": False,  # Note was overwritten
        "note_error": False,  # Error saving note
        "pdf_success": False,  # PDF downloaded successfully
        "pdf_existed": False,  # PDF already existed
        "pdf_overwritten": False,  # PDF was overwritten
        "pdf_error": False,  # Error downloading PDF
        "supp_success": False,  # Supplementary materials downloaded successfully
        "supp_existed": False,  # Supplementary materials already existed
        "supp_overwritten": False,  # Supplementary materials were overwritten
        "supp_error": False,  # Error downloading supplementary materials
    }

    # save note
    try:
        if is_save_notes:
            note_result = save_note(
                venue_path,
                note,
                additional_data={"details": note.details, "api_version": api_version},
                is_overwrite=is_overwrite_notes,
            )
            result["note_success"] = note_result.get("success", False)
            result["note_existed"] = note_result.get("existed", False)
            result["note_overwritten"] = note_result.get("overwritten", False)
    except Exception as e:
        result["note_error"] = True
        logger.error(f"Error saving note {note.id}: {str(e)}")

    # download pdf
    try:
        if is_save_pdfs:
            pdf_result = download_pdf_for_note(
                note,
                venue_path,
                client,
                api_version,
                is_overwrite=is_overwrite_pdfs,
            )
            result["pdf_success"] = pdf_result.get("success", False)
            result["pdf_existed"] = pdf_result.get("existed", False)
            result["pdf_overwritten"] = pdf_result.get("overwritten", False)
    except Exception as e:
        result["pdf_error"] = True
        logger.error(f"Error downloading PDF for note {note.id}: {str(e)}")

    # download supplementary
    try:
        if is_save_supplementaries:
            supp_result = download_supplementaries_material_for_note(
                note,
                venue_path,
                client,
                api_version,
                is_overwrite=is_overwrite_supplementaries,
            )
            result["supp_success"] = supp_result.get("success", False)
            result["supp_existed"] = supp_result.get("existed", False)
            result["supp_overwritten"] = supp_result.get("overwritten", False)
    except Exception as e:
        result["supp_error"] = True
        logger.error(f"Error downloading supplementary materials for note {note.id}: {str(e)}")

    return result
