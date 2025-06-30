from typing import Callable, List, Optional

from relepaper.domains.langgraph.entities.relevance_score import RelevanceScore


# %%
class RelevanceScoreContainer(List[RelevanceScore]):
    def __init__(self, scores: Optional[List[RelevanceScore]] = None):
        super().__init__(scores or [])

    def filter_by(self, key: str, func: Callable[[RelevanceScore], bool]) -> "RelevanceScoreContainer":
        return RelevanceScoreContainer([item for item in self if func(getattr(item, key))])

    def op_by(self, key: str, func: Callable[[List[float]], float]) -> Optional[float]:
        items = [getattr(item, key, None) for item in self]
        items = [item for item in items if item is not None]
        if not items:
            return None
        return func(items)

    @property
    def mean(self) -> float:
        # scores = [getattr(score, "score", 0) for score in self]
        # return sum(scores) / len(scores)
        return self.op_by("score", lambda x: sum(x) / len(x))


# %%
if __name__ == "__main__":
    container = RelevanceScoreContainer()
    container.append(RelevanceScore(score=0.9, criteria="test", comment="test"))
    container.append(
        RelevanceScore(score=0.85, criteria="test", comment="long comment with some text and some more text")
    )
    container.append(RelevanceScore(score=0.8, criteria="hello", comment="hello"))
    container.append(RelevanceScore(score=0.4, criteria="world", comment="world"))
    print(container)
    print(f"test: {container.filter_by('criteria', lambda x: x == 'test')}")
    print(f"hello: {container.filter_by('criteria', lambda x: x == 'hello')}")
    print(f"world: {container.filter_by('criteria', lambda x: x == 'world')}")  # []
    print(f"score >0.5: {container.filter_by('score', lambda x: x > 0.5)}")
    print(f"score <0.5: {container.filter_by('score', lambda x: x < 0.5)}")
    print(f"mean: {container.op_by('score', lambda x: sum(x) / len(x))}")
    print(f"max: {container.op_by('score', lambda x: max(x))}")
    print(f"min: {container.op_by('score', lambda x: min(x))}")
    print(f"len: {len(container)}")
    print(f"mean: {container.mean}")
    print(container[0])
    container[0] = RelevanceScore(score=0.95, criteria="omfg", comment="test")
    print(container[0])
    print(container)
