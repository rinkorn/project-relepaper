# %%
from pprint import pformat
from typing import Callable, List, Optional

from relepaper.domains.langgraph.entities.relevance_score import RelevanceCriteria, RelevanceScore, Score


# %%
class RelevanceScoreContainer(List[RelevanceScore]):
    def __init__(self, scores: Optional[List[RelevanceScore]] = None):
        super().__init__(scores or [])

    def filter_by(self, key: str, func: Callable[[RelevanceScore], bool]) -> "RelevanceScoreContainer":
        """
        Filter the container by the key and the function.
        """
        return RelevanceScoreContainer([item for item in self if func(getattr(item, key))])

    # def op_by(self, key: str, func: Callable[[List[float]], float]) -> Optional[float]:
    #     """
    #     Apply function to the list of values of the key.
    #     """
    #     items = [getattr(item, key, None) for item in self]
    #     items = [item for item in items if item is not None]
    #     return func(items)

    @property
    def mean(self) -> Optional[float]:
        """
        Calculate the mean of the scores.
        """
        return sum([item.score.value for item in self]) / len(self)

    def __str__(self) -> str:
        return f"{self.__class__.__name__}(\n" + pformat([str(item) for item in self], indent=1, width=200) + "\n)"

    def __repr__(self) -> str:
        return self.__str__()


# %%
if __name__ == "__main__":
    container = RelevanceScoreContainer(
        scores=[
            RelevanceScore(score=Score(0.9), criteria=RelevanceCriteria.THEME, comment="test"),
            RelevanceScore(
                score=Score(0.85),
                criteria=RelevanceCriteria.THEME,
                comment="long comment with some text and some more text",
            ),
            RelevanceScore(score=Score(0.8), criteria=RelevanceCriteria.THEME, comment="hello"),
            RelevanceScore(score=Score(0.4), criteria=RelevanceCriteria.THEME, comment="world"),
        ]
    )
    # print(container)
    print(f"test: {container.filter_by('criteria', lambda x: x == RelevanceCriteria.THEME)}")
    print(f"hello: {container.filter_by('criteria', lambda x: x == RelevanceCriteria.THEME)}")
    print(f"world: {container.filter_by('criteria', lambda x: x == RelevanceCriteria.THEME)}")
    print(f"score >0.5: {container.filter_by('score', lambda x: x.value > 0.5)}")
    print(f"score <0.5: {container.filter_by('score', lambda x: x.value < 0.5)}")
    print(f"mean value of scores: {container.mean}")
    print(f"len of container: {len(container)}")
    print(f"first item: {container[0]}")
    container[0] = RelevanceScore(score=Score(0.99), criteria=RelevanceCriteria.TERMINOLOGY, comment="test")
    print(f"first item after update: {container[0]}")
    # print(container)
