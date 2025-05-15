# %%
import logging
from pathlib import Path
from relepaper.store.openreviewnet.statistics.filter_notes import (
    filter_by_existing_pdf,
    filter_by_pdf_field,
    filter_by_replies_field,
    filter_by_responses_field,
)

from relepaper.store.openreviewnet.statistics.scan_files import (
    scan_notes_json_files,
    scan_pdf_files,
    scan_venue_json_files,
)

from relepaper.store.openreviewnet.statistics.extract_from_note import (
    extract_responses_from_note_json,
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
    notes, _ = filter_by_replies_field(notes_files)
    notes, _ = filter_by_pdf_field(notes)
    notes, _ = filter_by_existing_pdf(notes)
    notes, _ = filter_by_responses_field(notes, "Official_Review")
    notes, _ = filter_by_responses_field(notes, "Decision")
    print(
        "Количество <note>.json, с:\n",
        " * заполненными полями <pdf>\n",
        " * заполненными полями <replies>\n",
        " * существующими PDF файлами\n",
        " * присутствующими Official_Review\n",
        " * присутствующими Decision\n",
        len(notes),
    )

# %%
if __name__ == "__main__":
    # decisions statistics
    decisions_dict = {}
    for note in notes:
        official_reviews = extract_responses_from_note_json(note, "Official_Review")
        decisions = extract_responses_from_note_json(note, "Decision")
        decision = decisions[0]["content"]["decision"]
        if isinstance(decision, dict) and "value" in decision:
            decision = decision["value"]
        if isinstance(decision, str):
            decisions_dict[decision] = decisions_dict.get(decision, 0) + 1

    # sort by count reversed
    decisions_dict = dict(sorted(decisions_dict.items(), key=lambda item: item[1], reverse=True))
    # print as table
    max_decision_len = max(len(decision) for decision in decisions_dict.keys())
    max_count_len = max(len(str(count)) for count in decisions_dict.values())
    print(f"{'Decision':<{max_decision_len}} {'Count':>{max_count_len}}")
    for decision, count in decisions_dict.items():
        print(f"{decision:<{max_decision_len}} {count:>{max_count_len}}")
    print(f"{' Total':<{max_decision_len}} {sum(decisions_dict.values()):>{max_count_len}}")
    print()

    # sort by decision
    decisions_dict = dict(sorted(decisions_dict.items(), key=lambda item: item[0]))
    # print as table
    max_decision_len = max(len(decision) for decision in decisions_dict.keys())
    max_count_len = max(len(str(count)) for count in decisions_dict.values())
    print(f"{'Decision':<{max_decision_len}} {'Count':>{max_count_len}}")
    for decision, count in decisions_dict.items():
        print(f"{decision:<{max_decision_len}} {count:>{max_count_len}}")
    print(f"{' Total':<{max_decision_len}} {sum(decisions_dict.values()):>{max_count_len}}")
    print()


# %%
if __name__ == "__main__":
    import matplotlib.pyplot as plt

    # sort by count
    decisions_dict = dict(sorted(decisions_dict.items(), key=lambda item: item[1]))

    # plot bar chart horizontal
    fig, ax = plt.subplots(1, 1, figsize=(8, 14))
    ax.barh(decisions_dict.keys(), decisions_dict.values())
    # ax.set_xticks([d for d in decisions_dict.keys()])
    # ax.set_xticklabels([d for d in decisions_dict.keys()], rotation=90)
    ax.set_title("Decision keys statistics")
    ax.set_xlabel("Count")
    ax.set_ylabel("Decision key")
    fig.show()
    fig.savefig("decisions_horizontal.png", bbox_inches="tight")
    plt.close()


# %%
if __name__ == "__main__":
    # split keys by decision
    decisions_keys = decisions_dict.keys()
    decisions_keys = list(reversed(decisions_keys))
    print()

    # summary all accepted papers
    accepted_keys = [key for key in decisions_keys if "Accept" in key]
    decisions_keys = [key for key in decisions_keys if key not in accepted_keys]
    print(f"{'Decision (accepted)':<{max_decision_len}} {'Count':>{max_count_len}}")
    for accepted_key in accepted_keys:
        print(f"{accepted_key:<{max_decision_len}} {decisions_dict[accepted_key]:>{max_count_len}}")
    accepted_papers = sum([decisions_dict[key] for key in accepted_keys])
    print(f"{'  Количество принятых статей:':<{max_decision_len}} {accepted_papers:>{max_count_len}}")
    print()

    # summary all rejected papers
    rejected_keys = [key for key in decisions_keys if "Reject" in key]
    decisions_keys = [key for key in decisions_keys if key not in rejected_keys]
    print(f"{'Decision (rejected)':<{max_decision_len}} {'Count':>{max_count_len}}")
    for rejected_key in rejected_keys:
        print(f"{rejected_key:<{max_decision_len}} {decisions_dict[rejected_key]:>{max_count_len}}")
    rejected_papers = sum([decisions_dict[key] for key in rejected_keys])
    print(f"{'  Количество отклоненных статей:':<{max_decision_len}} {rejected_papers:>{max_count_len}}")
    print()

    # summary all revision papers
    revision_keys = [key for key in decisions_keys if "revis" in key.lower()]
    decisions_keys = [key for key in decisions_keys if key not in revision_keys]
    print(f"{'Decision (revision)':<{max_decision_len}} {'Count':>{max_count_len}}")
    for revision_key in revision_keys:
        print(f"{revision_key:<{max_decision_len}} {decisions_dict[revision_key]:>{max_count_len}}")
    revision_papers = sum([decisions_dict[key] for key in revision_keys])
    print(f"{'  Количество статей на доработку:':<{max_decision_len}} {revision_papers:>{max_count_len}}")
    print()

    # summary all remain papers
    remain_keys = decisions_keys
    decisions_keys = [key for key in decisions_keys if key not in revision_keys]
    print(f"{'Decision (remain)':<{max_decision_len}} {'Count':>{max_count_len}}")
    for remain_key in remain_keys:
        print(f"{remain_key:<{max_decision_len}} {decisions_dict[remain_key]:>{max_count_len}}")
    remain_papers = sum([decisions_dict[key] for key in remain_keys])
    print(f"{'  Количество остальных статей:':<{max_decision_len}} {remain_papers:>{max_count_len}}")
    print()


# %%
if __name__ == "__main__":
    # plot bar chart
    fig, ax = plt.subplots(1, 1, figsize=(8, 14))
    ax.barh(decisions_dict.keys(), decisions_dict.values())
    ax.set_title("Decision keys statistics")
    ax.set_xlabel("Count")
    ax.set_ylabel("Decision key")
    fig.savefig("decisions_horizontal.png", bbox_inches="tight")
    fig.show()
    plt.close()

# %%
