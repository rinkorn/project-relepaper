# %%
import uuid
from datetime import datetime
from typing import Annotated

from pydantic import BaseModel, Field


class Session(BaseModel):
    id: Annotated[str, "Session ID"] = Field(default_factory=lambda: uuid.uuid4().hex)
    created_at: Annotated[datetime, "Created at"] = Field(default_factory=datetime.now)
    updated_at: Annotated[datetime, "Updated at"] = Field(default_factory=datetime.now)


# %%
if __name__ == "__main__":
    session = Session()
    print(session)
    print(session.id)
    print(session.created_at)
    print(session.updated_at)
