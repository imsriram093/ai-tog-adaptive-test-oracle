from __future__ import annotations
import json
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class BaselineExample:
    text: str
    approved_by: str = "unspecified"
    notes: str = ""


@dataclass
class TestCategory:
    """
    Stage 1 artifact: a curated, QA-approved set of known-good answers for
    one test-case category, plus the structural/domain rules that apply to
    that category. This is the ground truth Stage 2 turns into a reference
    profile.
    """
    __test__ = False

    category_id: str

    prompt: str
    baseline_examples: list[BaselineExample]
    rejected_examples: list[BaselineExample] = field(default_factory=list)
    min_length: int | None = None
    max_length: int | None = None
    required_terms: list[str] = field(default_factory=list)
    banned_terms: list[str] = field(default_factory=list)

    def __post_init__(self):
        if len(self.baseline_examples) < 3:
            raise ValueError(
                f"Category '{self.category_id}' has only "
                f"{len(self.baseline_examples)} baseline examples; at least 3 "
                "are required to compute a meaningful centroid and spread."
            )


def load_category(path: str | Path) -> TestCategory:
    data = json.loads(Path(path).read_text())
    return TestCategory(
        category_id=data["category_id"],
        prompt=data["prompt"],
        baseline_examples=[BaselineExample(**e) for e in data["baseline_examples"]],
        rejected_examples=[BaselineExample(**e) for e in data.get("rejected_examples", [])],
        min_length=data.get("min_length"),
        max_length=data.get("max_length"),
        required_terms=data.get("required_terms", []),
        banned_terms=data.get("banned_terms", []),
    )


def load_categories(directory: str | Path) -> dict[str, TestCategory]:
    directory = Path(directory)
    categories = {}
    for path in sorted(directory.glob("*.json")):
        cat = load_category(path)
        categories[cat.category_id] = cat
    return categories
