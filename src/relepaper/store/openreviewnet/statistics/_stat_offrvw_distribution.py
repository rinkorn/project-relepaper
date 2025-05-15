# %%
import logging
from pathlib import Path

from relepaper.store.openreviewnet.statistics.extract_from_note import (
    extract_responses_from_note_json,
)
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
    # # count of decisions
    # count_of_decisions = {}
    # for note in notes:
    #     decisions = extract_responses_from_note_json(note, "Decision")
    #     count_of_decisions[len(decisions)] = count_of_decisions.get(len(decisions), 0) + 1
    # print(f"Количество decisions в: {count_of_decisions}")

    # count of official reviews
    count_of_official_reviews = {}
    for note in notes:
        official_reviews = extract_responses_from_note_json(note, "Official_Review")
        count_of_official_reviews[len(official_reviews)] = count_of_official_reviews.get(len(official_reviews), 0) + 1
    print(f"Количество official reviews в: {count_of_official_reviews}")


# %%
if __name__ == "__main__":
    # plot_count_of_official_reviews
    import matplotlib.pyplot as plt

    plt.figure(figsize=(10, 5))
    plt.bar(count_of_official_reviews.keys(), count_of_official_reviews.values())
    plt.xlabel("Количество official reviews на note")
    plt.xlim(0, 13)
    plt.xticks(range(0, 13))
    plt.ylabel("Количество notes")
    plt.title("Количество notes с определенным количеством official reviews")
    plt.savefig("count_of_notes_with_official_reviews.png")
    plt.show()
    plt.close()

# %%
