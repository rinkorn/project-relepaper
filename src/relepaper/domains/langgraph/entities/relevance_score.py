# %%
import json
from dataclasses import dataclass
from enum import Enum


# %%
@dataclass(frozen=True)
class Score:
    """Score value between min_value and max_value"""

    value: float | int
    min_value: float | int = 0.0
    max_value: float | int = 100.0

    def __post_init__(self):
        if self.min_value > self.max_value:
            raise ValueError(f"Min value must be less than max value: {self.min_value} > {self.max_value}")
        if self.value < self.min_value or self.value > self.max_value:
            raise ValueError(f"Score value must be between {self.min_value} and {self.max_value}")

    def __str__(self) -> str:
        # return f"'Score(value={self.value:.2f})'"
        return f"{self.__class__.__name__}({json.dumps(self.__dict__, indent=1)})"

    def __repr__(self) -> str:
        return self.__str__()


if __name__ == "__main__":
    score0 = Score(value=2.0)
    print(score0)
    print(json.dumps(score0.__dict__, indent=1))
    score1 = Score(value=90.0)
    score2 = Score(value=85.0)
    score3 = Score(value=90.0)

    scores = [score0, score1, score2, score3]

    print(score1 == score3)  # True
    print(score1 == score2)  # False

    print(f"sum_scores: {sum([score.value for score in scores])}")
    print(f"mean_score: {sum([score.value for score in scores]) / len(scores)}")


# %%
class RelevanceCriteria(Enum):
    """Criteria for relevance score"""

    THEME = "theme"
    TERMINOLOGY = "terminology"
    METHODOLOGY = "methodology"
    PRACTICAL_APPLICABILITY = "practical applicability"
    NOVELTY_AND_RELEVANCE = "novelty and relevance"
    FUNDAMENTAL_SIGNIFICANCE = "fundamental significance"


# %%
@dataclass
class RelevanceScore:
    score: Score
    criteria: RelevanceCriteria
    comment: str | None = None

    def __str__(self) -> str:
        return f"{self.__class__.__name__}(criteria=Criteria.{self.criteria.value}, score=Score({self.score.value}), comment={self.comment if self.comment else 'None'})"

    def __repr__(self) -> str:
        return self.__str__()


if __name__ == "__main__":
    relevance_score = RelevanceScore(
        score=Score(value=90),
        criteria=RelevanceCriteria.THEME,
        comment="simple comment about the relevance score",
    )
    print(relevance_score)
