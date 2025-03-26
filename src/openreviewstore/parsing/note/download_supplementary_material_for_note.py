# %%
import logging

logger = logging.getLogger(__name__)


def download_supplementary_material_for_note(client, note):
    if note.content.get("supplementary_material", {}).get("value"):
        f = client.get_attachment(note.id, "supplementary_material")
        file_name = f"pdfs/{note.id}-{note.number}_supplementary_material.zip"
        with open(file_name, "wb") as op:
            op.write(f)
