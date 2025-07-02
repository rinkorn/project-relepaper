import json
from dataclasses import dataclass
from enum import Enum


# %%
class RepeatStatus(Enum):
    REPEAT = "REPEAT"
    STOP = "STOP"


@dataclass(frozen=True)
class RepeatDecision:
    status: RepeatStatus
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


# %%
if __name__ == "__main__":
    repeat_decision = RepeatDecision(
        status=RepeatStatus.REPEAT,
        comment="simple comment about the decision",
    )
    print(repeat_decision)
    print(repeat_decision.status)
    print(repeat_decision.comment)
