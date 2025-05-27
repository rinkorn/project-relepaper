# %%
import logging
from pathlib import Path

from relepaper.store.openreviewnet.statistics.filter_notes import (
    filter_by_existing_pdf,
    filter_by_responses_field,
)
from relepaper.store.openreviewnet.statistics.scan_files import (
    scan_notes_json_files,
    scan_pdf_files,
    scan_venue_json_files,
)

logger = logging.getLogger(__name__)


# %%
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # store_path = Path("/home/rinkorn/space/prog/python/sber/project-openreviewstore/data/store/")
    store_path = Path("/data/data.sets/openreviewstore/")

    # scanning
    venue_files = scan_venue_json_files(store_path)
    print(f"Количество обработанных venue.json файлов: {len(venue_files)}")

    notes_files = scan_notes_json_files(store_path)
    print(f"Количество загруженных <note>.json файлов: {len(notes_files)}")

    pdf_files = scan_pdf_files(store_path)
    print(f"Количество загруженных <pdf>.pdf файлов: {len(pdf_files)}")

    # filtering
    # notes, _ = filter_by_replies_field(notes_files)
    # notes, _ = filter_by_pdf_field(notes_files)
    notes, _ = filter_by_existing_pdf(notes_files)
    notes, _ = filter_by_responses_field(notes, "Official_Review")
    notes, _ = filter_by_responses_field(notes, "Decision")
    # " * заполненными полями <replies>\n",
    # " * заполненными полями <pdf>\n",
    print(
        "Количество <note>.json, с:\n",
        " * существующими PDF файлами\n",
        " * присутствующим[и] Official_Review\n",
        " * присутствующим Decision\n",
        len(notes),
    )

# %%
if __name__ == "__main__":
    pass


# %%
