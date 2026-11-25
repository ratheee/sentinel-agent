"""A small inventory module with a few bugs, used as a demo task for Sentinel."""

from dataclasses import dataclass


@dataclass
class Item:
    sku: str
    name: str
    unit_price: float
    quantity: int = 0


class Inventory:
    def __init__(self, low_stock_threshold: int = 5):
        self.items: dict[str, Item] = {}
        self.low_stock_threshold = low_stock_threshold

    def add_item(self, sku: str, name: str, unit_price: float, quantity: int = 0) -> Item:
        if unit_price < 0:
            raise ValueError("unit_price must be non-negative")
        if sku in self.items:
            self.items[sku].quantity += quantity
        else:
            self.items[sku] = Item(sku, name, unit_price, quantity)
        return self.items[sku]

    def remove_stock(self, sku: str, quantity: int) -> int:
        item = self.items[sku]
        item.quantity -= quantity
        return item.quantity

    def total_value(self) -> float:
        return round(sum(item.unit_price for item in self.items.values()), 2)

    def low_stock(self) -> list[str]:
        return sorted(sku for sku, item in self.items.items() if item.quantity > self.low_stock_threshold)

    def apply_discount(self, sku: str, percent: float) -> float:
        item = self.items[sku]
        item.unit_price = round(item.unit_price * percent / 100, 2)
        return item.unit_price
