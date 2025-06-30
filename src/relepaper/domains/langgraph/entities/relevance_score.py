import json
from dataclasses import dataclass
from pprint import pformat


@dataclass
class RelevanceScore:
    score: float
    criteria: str
    comment: str = ""

    def __str__(self) -> str:
        return pformat({self.criteria: self.score})

    def __repr__(self) -> str:
        return self.__str__()

    def to_dict(self) -> dict:
        return self.__dict__

    @classmethod
    def from_dict(cls, data: dict) -> "RelevanceScore":
        return cls(**data)

    @classmethod
    def from_json(cls, json_str: str) -> "RelevanceScore":
        return cls.from_dict(json.loads(json_str))

    def to_json(self) -> str:
        return json.dumps(self.to_dict())


if __name__ == "__main__":
    relevance_score = RelevanceScore(score=0.9, criteria="test", comment="test")
    print(relevance_score)
    # print(relevance_score.to_dict())
    # print(relevance_score.to_json())
    # print(RelevanceScore.from_json(relevance_score.to_json()))
