from typing import TypedDict


# %%
class RelevanceDecision(TypedDict):
    decision: str
    comment: str | None = None


# %%
if __name__ == "__main__":
    decision = RelevanceDecision(decision="good", comment="good")
    print(decision)
