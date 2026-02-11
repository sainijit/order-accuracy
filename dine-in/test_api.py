#!/usr/bin/env python3
"""
Test script for Dine-In Order Accuracy API
"""

import requests
import json
from pathlib import Path


# API Configuration
BASE_URL = "http://localhost:8083"
API_BASE = f"{BASE_URL}/api"


def test_health_check():
    """Test health endpoint"""
    print("\n=== Testing Health Check ===")
    response = requests.get(f"{BASE_URL}/health")
    print(f"Status: {response.status_code}")
    print(f"Response: {response.json()}")
    return response.status_code == 200


def test_validate_single():
    """Test single plate validation"""
    print("\n=== Testing Single Plate Validation ===")
    
    # Prepare order data
    order_data = {
        "order_id": "image_01_mcd_combo",
        "table_number": "5",
        "restaurant": "McDonald's",
        "items": [
            {"name": "Big Mac", "quantity": 1},
            {"name": "French Fries", "quantity": 1},
            {"name": "Coca-Cola", "quantity": 1}
        ],
        "modifiers": ["No pickles"]
    }
    
    # Check if image exists
    image_path = Path("images/image_01_mcd_combo.png")
    if not image_path.exists():
        print(f"Warning: Image not found at {image_path}")
        return False
    
    # Prepare multipart form data
    files = {
        'image': ('plate.png', open(image_path, 'rb'), 'image/png')
    }
    data = {
        'order': json.dumps(order_data)
    }
    
    # Make request
    response = requests.post(f"{API_BASE}/validate", files=files, data=data)
    print(f"Status: {response.status_code}")
    
    if response.status_code == 200:
        result = response.json()
        print(f"Validation ID: {result['validation_id']}")
        print(f"Order Complete: {result['order_complete']}")
        print(f"Accuracy Score: {result['accuracy_score']}")
        print(f"Missing Items: {result['missing_items']}")
        print(f"Extra Items: {result['extra_items']}")
        return True, result['validation_id']
    else:
        print(f"Error: {response.text}")
        return False, None


def test_get_validation_result(validation_id):
    """Test retrieving validation result"""
    print(f"\n=== Testing Get Validation Result ===")
    response = requests.get(f"{API_BASE}/validate/{validation_id}")
    print(f"Status: {response.status_code}")
    
    if response.status_code == 200:
        result = response.json()
        print(f"Retrieved Validation ID: {result['validation_id']}")
        print(f"Timestamp: {result['timestamp']}")
        return True
    else:
        print(f"Error: {response.text}")
        return False


def test_list_validations():
    """Test listing all validations"""
    print("\n=== Testing List All Validations ===")
    response = requests.get(f"{API_BASE}/validate")
    print(f"Status: {response.status_code}")
    
    if response.status_code == 200:
        result = response.json()
        print(f"Total Validations: {result['total']}")
        return True
    else:
        print(f"Error: {response.text}")
        return False


def test_batch_validation():
    """Test batch validation"""
    print("\n=== Testing Batch Validation ===")
    
    # Prepare orders data for multiple images
    orders_data = [
        {
            "image_id": "image_01_mcd_combo",
            "order_id": "order_001",
            "table_number": "5",
            "restaurant": "McDonald's",
            "items": [
                {"name": "Big Mac", "quantity": 1},
                {"name": "French Fries", "quantity": 1}
            ]
        },
        {
            "image_id": "image_02_mcd_variation",
            "order_id": "order_002",
            "table_number": "7",
            "restaurant": "McDonald's",
            "items": [
                {"name": "Chicken Nuggets", "quantity": 1},
                {"name": "French Fries", "quantity": 1}
            ]
        }
    ]
    
    # Check images
    image_paths = [
        Path("images/image_01_mcd_combo.png"),
        Path("images/image_02_mcd_variation.png")
    ]
    
    files = []
    for img_path in image_paths:
        if img_path.exists():
            files.append(('images', (img_path.name, open(img_path, 'rb'), 'image/png')))
        else:
            print(f"Warning: Image not found at {img_path}")
    
    if len(files) != len(orders_data):
        print("Skipping batch test - not all images available")
        return False
    
    data = {
        'orders': json.dumps(orders_data)
    }
    
    response = requests.post(f"{API_BASE}/validate/batch", files=files, data=data)
    print(f"Status: {response.status_code}")
    
    if response.status_code == 200:
        result = response.json()
        print(f"Batch Results: {len(result['results'])} validations")
        for idx, res in enumerate(result['results']):
            if 'validation_id' in res and res['validation_id']:
                print(f"  [{idx+1}] ID: {res['validation_id']}, Accuracy: {res.get('accuracy_score', 'N/A')}")
        return True
    else:
        print(f"Error: {response.text}")
        return False


def main():
    """Run all tests"""
    print("=" * 60)
    print("Dine-In Order Accuracy API Tests")
    print("=" * 60)
    
    results = {}
    
    # Test health check
    results['health'] = test_health_check()
    
    # Test single validation
    success, validation_id = test_validate_single()
    results['validate_single'] = success
    
    # Test get validation result
    if validation_id:
        results['get_result'] = test_get_validation_result(validation_id)
    
    # Test list validations
    results['list_validations'] = test_list_validations()
    
    # Test batch validation
    results['batch_validation'] = test_batch_validation()
    
    # Summary
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)
    for test_name, passed in results.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{status} - {test_name}")
    
    total_passed = sum(1 for v in results.values() if v)
    print(f"\nTotal: {total_passed}/{len(results)} tests passed")
    print("=" * 60)


if __name__ == "__main__":
    main()
