from pydantic import BaseModel, Field
from llm.registry import ToolRegistry

# Initialize Registry
registry = ToolRegistry()

# --- Request Schemas ---

class AddToCartSchema(BaseModel):
    product_id: str = Field(description="The unique identifier of the product to add.")
    quantity: int = Field(description="The quantity of the product to add.", default=1)

class TrackOrderSchema(BaseModel):
    order_id: str = Field(description="The unique identifier of the order to track.")

class PlaceOrderSchema(BaseModel):
    payment_method: str = Field(description="The payment method (e.g., 'credit_card', 'paypal').")
    shipping_address: str = Field(description="The full shipping address.")

# --- Tool Implementations ---

@registry.register_function(
    name="add_to_cart", 
    description="Add a product to the shopping cart.", 
    args_schema=AddToCartSchema
)
def add_to_cart(product_id: str, quantity: int = 1) -> str:
    """Mock implementation to add an item to the cart."""
    print(f"[TOOL] Adding {quantity} of {product_id} to cart.")
    return f"Successfully added {quantity} unit(s) of product '{product_id}' to your cart."

@registry.register_function(
    name="track_order",
    description="Get the current status of an order.",
    args_schema=TrackOrderSchema
)
def track_order(order_id: str) -> str:
    """Mock implementation to track an order status."""
    statuses = {
        "123": "Shipped - Arriving tomorrow",
        "456": "Processing",
        "789": "Delivered"
    }
    status = statuses.get(order_id, "Order not found. Please check the ID.")
    print(f"[TOOL] Tracking order {order_id}: {status}")
    return f"Order status for {order_id}: {status}"

@registry.register_function(
    name="place_order",
    description="Place an order with payment and shipping details.",
    args_schema=PlaceOrderSchema
)
def place_order(payment_method: str, shipping_address: str) -> str:
    """Mock implementation to place an order."""
    import uuid
    new_order_id = str(uuid.uuid4())[:8]
    print(f"[TOOL] Placing order to {shipping_address} via {payment_method}.")
    return f"Order placed successfully! Your order ID is {new_order_id}. Estimated delivery: 3-5 business days."
