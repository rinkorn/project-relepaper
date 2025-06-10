import json
import logging
from pathlib import Path
from typing import List, Optional

from relepaper.domains.openalex.entities.work import OpenAlexWork
from relepaper.domains.openalex.external.interfaces import IRepository

logger = logging.getLogger(__name__)


class OnFileSystemWorksRepository(IRepository):
    """
    Repository that saves OpenAlex works to the file system.

    Each work is saved as a separate JSON file named by work ID.
    """

    def __init__(self, storage_path: Path):
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        self._logger = logger

    def get_by_id(self, clean_id: str) -> Optional[OpenAlexWork]:
        """Get work by ID from file system."""
        if not clean_id:
            return None

        work_path = self.storage_path / f"{clean_id}.json"
        if not work_path.exists():
            self._logger.debug(f"Work file not found: {work_path}")
            return None

        try:
            with open(work_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            # Ensure ID is set correctly
            data["id"] = "https://openalex.org/" + clean_id
            return OpenAlexWork.from_dict(data)

        except (json.JSONDecodeError, KeyError, ValueError) as e:
            self._logger.error(f"Failed to load work {clean_id}: {e}")
            return None

    def get_by_ids(self, ids: List[str]) -> List[OpenAlexWork]:
        """Get works by IDs from file system."""
        return [self.get_by_id(id) for id in ids if self.exists(id)]

    def save(self, work: OpenAlexWork) -> None:
        """Save work to file system."""
        if not work.id:
            raise ValueError("Work ID is required for saving")

        work_path = self.storage_path / f"{work.clean_id}.json"

        try:
            with open(work_path, "w", encoding="utf-8") as f:
                json.dump(work.to_dict(), f, indent=2, ensure_ascii=False)
            self._logger.debug(f"Saved work: {work_path}")

        except Exception as e:
            self._logger.error(f"Failed to save work {work.clean_id}: {e}")
            raise

    def save_all(self, works: List[OpenAlexWork]) -> None:
        """Save all works to file system."""
        for work in works:
            self.save(work)

    def delete(self, clean_id: str) -> None:
        """Delete work from file system."""
        if not clean_id:
            return

        work_path = self.storage_path / f"{clean_id}.json"

        try:
            if work_path.exists():
                work_path.unlink()
                self._logger.debug(f"Deleted work: {work_path}")
            else:
                self._logger.warning(f"Work file not found for deletion: {work_path}")

        except Exception as e:
            self._logger.error(f"Failed to delete work {clean_id}: {e}")
            raise

    def delete_all(self) -> None:
        """Delete all works from file system."""
        for work_path in self.storage_path.glob("*.json"):
            work_path.unlink()
            self._logger.debug(f"Deleted work: {work_path}")

    def exists(self, clean_id: str) -> bool:
        """Check if work exists in storage."""
        if not clean_id:
            return False
        return (self.storage_path / f"{clean_id}.json").exists()

    def list_all_ids(self) -> List[str]:
        """Get list of all work IDs in storage."""
        return [path.stem for path in self.storage_path.glob("*.json") if path.is_file()]

    def count(self) -> int:
        """Get total count of works in storage."""
        return len(list(self.storage_path.glob("*.json")))


if __name__ == "__main__":
    import uuid
    from datetime import datetime

    from relepaper.config.dev_settings import get_dev_settings

    # Test the repository
    session_id = uuid.uuid4().hex
    datetime_str = datetime.now().strftime("%Y%m%dT%H%M%S")
    storage_path = get_dev_settings().project_path / "session" / f"test_{datetime_str}_{session_id}" / "openalex_works"
    repository = OnFileSystemWorksRepository(storage_path)

    # Create test work
    work_id = f"https://openalex.org/W{uuid.uuid4().hex}"
    work1 = OpenAlexWork(id=work_id, title="Test Work", doi="10.1000/test")

    # Test operations
    repository.save(work1)
    loaded_work = repository.get_by_id(work1.clean_id)
    print(f"Loaded work: {loaded_work}")

    print(f"Total works: {repository.count()}")
    print(f"All IDs: {repository.list_all_ids()}")

    # Test delete
    repository.delete(work1.clean_id)
    print(f"Total works: {repository.count()}")
    print(f"All IDs: {repository.list_all_ids()}")

    # Test save all
    work2 = OpenAlexWork(id=f"https://openalex.org/W{uuid.uuid4().hex}", title="Test Work 2", doi="10.1000/test2")
    work3 = OpenAlexWork(id=f"https://openalex.org/W{uuid.uuid4().hex}", title="Test Work 3", doi="10.1000/test3")
    repository.save_all([work2, work3])
    print(f"Total works: {repository.count()}")
    print(f"All IDs: {repository.list_all_ids()}")

    # Test get by ids
    loaded_works = repository.get_by_ids([work1.clean_id, work2.clean_id, work3.clean_id])
    print(f"Loaded works: {[w.clean_id for w in loaded_works]}")

    # Test delete all
    repository.delete_all()
    print(f"Total works: {repository.count()}")
    print(f"All IDs: {repository.list_all_ids()}")
