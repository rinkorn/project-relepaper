# %%
import json
from dataclasses import dataclass
from enum import Enum


# %%
class RelevanceStatus(Enum):
    RELEVANT = "RELEVANT"
    NOT_RELEVANT = "NOT_RELEVANT"


# %%
@dataclass(frozen=True)
class Threshold:
    value: float
    min_value: float = 0.0
    max_value: float = 100.0

    def __post_init__(self):
        if self.value < self.min_value or self.value > self.max_value:
            raise ValueError(f"Threshold value must be between {self.min_value} and {self.max_value}")

    def __str__(self) -> str:
        return f"{self.__class__.__name__}({
            json.dumps(
                {
                    'value': self.value,
                    'min_value': self.min_value,
                    'max_value': self.max_value,
                },
                indent=1,
            )
        })"

    def __repr__(self) -> str:
        return self.__str__()


if __name__ == "__main__":
    threshold = Threshold(value=40.0)
    print(threshold)


# %%
@dataclass(frozen=True)
class RelevanceDecision:
    status: RelevanceStatus
    comment: str | None = None

    def __str__(self) -> str:
        return f"{self.__class__.__name__}({
            json.dumps(
                {
                    'status': self.status.value,
                    'comment': self.comment,
                },
                indent=1,
            )
        })"

    def __repr__(self) -> str:
        return self.__str__()


if __name__ == "__main__":
    decision = RelevanceDecision(
        status=RelevanceStatus.RELEVANT,
        comment="decicion computed with score and threshold of 0 to 100",
    )
    print(decision)
