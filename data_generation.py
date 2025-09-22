# data_generation.py
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random

# Ensure data directory exists
os.makedirs("data", exist_ok=True)

def generate_restaurant_data(num_samples=10000, output_path="data/restaurant_orders.csv"):
    """Generate synthetic restaurant order data and save to CSV."""
    np.random.seed(42)
    random.seed(42)

    cuisines = ['Italian', 'Chinese', 'Mexican', 'Indian', 'Japanese',
                'American', 'Thai', 'French', 'Mediterranean', 'Korean']
    meal_types = ['Breakfast', 'Lunch', 'Dinner', 'Snack']
    payment_methods = ['Credit Card', 'Debit Card', 'PayPal', 'Cash on Delivery', 'Digital Wallet']
    order_statuses = ['Delivered', 'Cancelled', 'Late Delivery', 'Issues with Order']
    countries = ['USA', 'UK', 'Canada', 'Australia', 'Germany', 'France', 'Japan', 'India', 'Brazil', 'Mexico']

    end_date = datetime.now()
    start_date = end_date - timedelta(days=180)
    date_range = [start_date + timedelta(days=x) for x in range(0, 181)]

    data = []
    for i in range(num_samples):
        order_date = random.choice(date_range)
        order_time = random.randint(10, 22)  # 10 AM to 10 PM

        cuisine = random.choice(cuisines)
        meal_type = random.choice(meal_types)
        payment_method = random.choice(payment_methods)
        country = random.choice(countries)

        if cuisine in ['Japanese', 'French']:
            order_value = round(np.random.normal(45, 15), 2)
        elif cuisine in ['Italian', 'Mediterranean']:
            order_value = round(np.random.normal(35, 12), 2)
        else:
            order_value = round(np.random.normal(25, 10), 2)

        order_value = max(5, order_value)  # minimum order value

        delivery_distance = round(np.random.exponential(5), 2)
        prep_time = round(np.random.normal(30, 10), 2)
        prep_time = max(10, prep_time)
        num_items = random.randint(1, 8)
        is_weekend = 1 if order_date.weekday() >= 5 else 0
        is_peak_hour = 1 if (12 <= order_time <= 14) or (18 <= order_time <= 20) else 0

        if delivery_distance > 10 and random.random() > 0.3:
            status = 'Late Delivery'
        elif order_value > 70 and random.random() > 0.6:
            status = 'Cancelled'
        elif prep_time > 40 and random.random() > 0.4:
            status = 'Issues with Order'
        else:
            status = 'Delivered'

        if random.random() < 0.1:
            status = random.choice(order_statuses)

        data.append([
            i + 1,
            order_date.strftime('%Y-%m-%d'),
            order_time,
            cuisine,
            meal_type,
            payment_method,
            country,
            order_value,
            delivery_distance,
            prep_time,
            num_items,
            is_weekend,
            is_peak_hour,
            status
        ])

    columns = [
        'order_id', 'order_date', 'order_time', 'cuisine', 'meal_type',
        'payment_method', 'country', 'order_value', 'delivery_distance',
        'prep_time', 'num_items', 'is_weekend', 'is_peak_hour', 'status'
    ]

    df = pd.DataFrame(data, columns=columns)
    df.to_csv(output_path, index=False)
    print(f"Generated {num_samples} samples and saved to {output_path}")
    return df

if __name__ == "__main__":
    generate_restaurant_data(10000)
