"""Test fixtures for model tests."""

from uuid import uuid4

import pytest

from bead.items.models import Item


@pytest.fixture
def test_items():
    """Create test 2AFC items.

    Returns
    -------
    callable
        Function that creates n test items.
    """

    def _create_items(n: int = 10) -> list[Item]:
        """Create n test items with 2AFC structure.

        Parameters
        ----------
        n : int
            Number of items to create.

        Returns
        -------
        list[Item]
            List of test items.
        """
        items = []
        for i in range(n):
            item = Item(
                item_template_id=uuid4(),
                rendered_elements={
                    "option_a": f"The child played with toy {i}. This is a natural sentence.",
                    "option_b": f"The person walked to place {i}. This is also natural.",
                },
                item_metadata={"test_index": i},
            )
            items.append(item)
        return items

    return _create_items


@pytest.fixture
def varied_test_items():
    """Create test items with varied content for better discrimination.

    Returns
    -------
    callable
        Function that creates n varied test items.
    """

    def _create_items(n: int = 10) -> list[Item]:
        """Create n varied test items.

        Parameters
        ----------
        n : int
            Number of items to create.

        Returns
        -------
        list[Item]
            List of varied test items.
        """
        # Create items with more varied content for better model discrimination
        templates = [
            ("The cat sat on the mat.", "The dog ran in the park."),
            ("She quickly finished her homework.", "He slowly ate his dinner."),
            (
                "The scientist discovered a new element.",
                "The artist painted a beautiful landscape.",
            ),
            ("Children love playing in the snow.", "Adults enjoy reading good books."),
            (
                "The movie was incredibly entertaining.",
                "The concert was surprisingly disappointing.",
            ),
            ("I need to buy groceries today.", "We should visit the museum tomorrow."),
            (
                "The weather forecast predicts rain.",
                "The news report confirmed the results.",
            ),
            (
                "She speaks three languages fluently.",
                "He plays two instruments professionally.",
            ),
            (
                "The company announced major layoffs.",
                "The team celebrated their victory.",
            ),
            ("My computer crashed unexpectedly.", "Her phone battery died suddenly."),
        ]

        items = []
        for i in range(n):
            template_idx = i % len(templates)
            option_a, option_b = templates[template_idx]

            item = Item(
                item_template_id=uuid4(),
                rendered_elements={
                    "option_a": f"{option_a} (variant {i})",
                    "option_b": f"{option_b} (variant {i})",
                },
                item_metadata={"test_index": i, "template": template_idx},
            )
            items.append(item)
        return items

    return _create_items
