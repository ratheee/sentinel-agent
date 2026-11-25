import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from inventory import Inventory  # noqa: E402


@pytest.fixture
def inv():
    inv = Inventory(low_stock_threshold=5)
    inv.add_item("A1", "Widget", 2.50, 10)
    inv.add_item("B2", "Gadget", 10.00, 3)
    inv.add_item("C3", "Doohickey", 1.25, 5)
    return inv


def test_add_existing_sku_increases_quantity(inv):
    inv.add_item("A1", "Widget", 2.50, 5)
    assert inv.items["A1"].quantity == 15


def test_total_value_accounts_for_quantity(inv):
    assert inv.total_value() == 61.25  # 25.00 + 30.00 + 6.25


def test_low_stock_includes_items_at_or_below_threshold(inv):
    assert inv.low_stock() == ["B2", "C3"]


def test_remove_stock_cannot_go_negative(inv):
    with pytest.raises(ValueError):
        inv.remove_stock("B2", 4)
    assert inv.items["B2"].quantity == 3


def test_remove_stock_rejects_non_positive_amounts(inv):
    with pytest.raises(ValueError):
        inv.remove_stock("A1", 0)


def test_discount_reduces_price_by_percentage(inv):
    assert inv.apply_discount("B2", 20) == 8.00


def test_discount_must_be_between_0_and_100(inv):
    with pytest.raises(ValueError):
        inv.apply_discount("A1", 150)
