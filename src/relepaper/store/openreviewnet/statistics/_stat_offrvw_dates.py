# %%
import logging
from datetime import datetime
from pathlib import Path

from relepaper.store.openreviewnet.statistics.extract_from_note import (
    extract_dates_from_reply,
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
def print_stat(stat_dict, sort_by="key"):
    # official reviews statistics by content keys
    if sort_by == "key":
        stat_dict = dict(sorted(stat_dict.items(), key=lambda item: item[0]))
    elif sort_by == "count":
        stat_dict = dict(sorted(stat_dict.items(), key=lambda item: item[1], reverse=True))
    max_key_len = max(len(key) for key in stat_dict.keys())
    max_count_len = max(len(str(count)) for count in stat_dict.values())
    for key, value in stat_dict.items():
        print(f"{key:<{max_key_len}} {value:>{max_count_len}}")
    print(f"{' Total keys:':<{max_key_len}} {len(stat_dict):>{max_count_len}}")
    print()


# %%
if __name__ == "__main__":
    # official reviews statistics by content keys and value begins with a number
    content_stat = {}
    for note in notes:
        official_reviews = extract_responses_from_note_json(note, "Official_Review")
        for review in official_reviews:
            content = review["content"]
            if isinstance(content, dict) and "value" in content:
                content = content["value"]
            for key, value in content.items():
                if isinstance(value, dict) and "value" in value:
                    value = value["value"]
                key = key.lower()
                content_stat[key] = content_stat.get(key, 0) + 1

    print("All keys that appear in official reviews and their count:")
    print_stat(content_stat, sort_by="count")

# %%
if __name__ == "__main__":
    # official reviews statistics by content keys and value begins with a number
    content_stat = {}
    cdate_stat = {}
    tcdate_stat = {}
    mdate_stat = {}
    tmdate_stat = {}
    ddate_stat = {}
    for note in notes:
        official_reviews = extract_responses_from_note_json(note, "Official_Review")
        for review in official_reviews:
            content = review["content"]
            if isinstance(content, dict) and "value" in content:
                content = content["value"]
            # if not "confidence" in content:
            #     continue
            # if not "rating" in content:
            #     continue
            # if ("summary" not in content) and ("review" not in content):
            #     continue
            for key, value in content.items():
                if isinstance(value, dict) and "value" in value:
                    value = value["value"]
                key = key.lower()
                content_stat[key] = content_stat.get(key, 0) + 1

            dates = extract_dates_from_reply(review)
            if dates["cdate"] is not None:
                cdate = datetime(*dates["cdate"].timetuple()[:2] + (1,))
                cdate_stat[cdate] = cdate_stat.get(cdate, 0) + 1
            if dates["tcdate"] is not None:
                tcdate = datetime(*dates["tcdate"].timetuple()[:2] + (1,))
                tcdate_stat[tcdate] = tcdate_stat.get(tcdate, 0) + 1
            if dates["mdate"] is not None:
                mdate = datetime(*dates["mdate"].timetuple()[:2] + (1,))
                mdate_stat[mdate] = mdate_stat.get(mdate, 0) + 1
            if dates["tmdate"] is not None:
                tmdate = datetime(*dates["tmdate"].timetuple()[:2] + (1,))
                tmdate_stat[tmdate] = tmdate_stat.get(tmdate, 0) + 1
            if dates["ddate"] is not None:
                ddate = datetime(*dates["ddate"].timetuple()[:2] + (1,))
                ddate_stat[ddate] = ddate_stat.get(ddate, 0) + 1

    print("All keys that appear in official reviews and their count:")
    print_stat(content_stat, sort_by="count")

# %%
if __name__ == "__main__":
    import matplotlib.pyplot as plt

    plt.figure(figsize=(8, 4))
    plt.bar(cdate_stat.keys(), cdate_stat.values(), width=26, align="center")
    plt.title(f"The creation date of the official reviews [sum: {sum(cdate_stat.values())}]")
    plt.xlabel("cdate")
    plt.ylabel("count of official reviews")
    plt.xlim(datetime(2017, 9, 1), datetime(2025, 6, 1))
    plt.gcf().subplots_adjust(bottom=0.2)
    plt.tight_layout()
    plt.savefig("offrvw_the_creation_date_of_the_official_reviews.png")
    plt.show()
    plt.close()

    plt.figure(figsize=(8, 4))
    plt.bar(tcdate_stat.keys(), tcdate_stat.values(), width=26, align="center")
    plt.title(f"The true creation date of the official reviews [sum: {sum(tcdate_stat.values())}]")
    plt.xlabel("tcdate")
    plt.ylabel("count of official reviews")
    plt.xlim(datetime(2017, 9, 1), datetime(2025, 6, 1))
    plt.gcf().subplots_adjust(bottom=0.2)
    plt.tight_layout()
    plt.savefig("offrvw_the_true_creation_date_of_the_official_reviews.png")
    plt.show()
    plt.close()

    plt.figure(figsize=(8, 4))
    plt.bar(mdate_stat.keys(), mdate_stat.values(), width=26, align="center")
    plt.title(f"The modification date of the official reviews [sum: {sum(mdate_stat.values())}]")
    plt.xlabel("mdate")
    plt.ylabel("count of official reviews")
    plt.xlim(datetime(2017, 9, 1), datetime(2025, 6, 1))
    plt.gcf().subplots_adjust(bottom=0.2)
    plt.tight_layout()
    plt.savefig("offrvw_the_modification_date_of_the_official_reviews.png")
    plt.show()
    plt.close()

    plt.figure(figsize=(8, 4))
    plt.bar(tmdate_stat.keys(), tmdate_stat.values(), width=26, align="center")
    plt.title(f"The true modification date of the official reviews [sum: {sum(tmdate_stat.values())}]")
    plt.xlabel("tmdate")
    plt.ylabel("count of official reviews")
    plt.xlim(datetime(2017, 9, 1), datetime(2025, 6, 1))
    plt.gcf().subplots_adjust(bottom=0.2)
    plt.tight_layout()
    plt.savefig("offrvw_the_true_modification_date_of_the_official_reviews.png")
    plt.show()
    plt.close()

    plt.figure(figsize=(8, 4))
    plt.bar(ddate_stat.keys(), ddate_stat.values(), width=26, align="center")
    plt.title(f"The deletion date of the official reviews [sum: {sum(ddate_stat.values())}]")
    plt.xlabel("ddate")
    plt.ylabel("count of official reviews")
    plt.xlim(datetime(2017, 9, 1), datetime(2025, 6, 1))
    plt.gcf().subplots_adjust(bottom=0.2)
    plt.tight_layout()
    plt.savefig("offrvw_the_deletion_date_of_the_official_reviews.png")
    plt.show()
    plt.close()

# %%
