# Script to create test users with hashed passwords in users.json

import json
import os
import bcrypt

# Path to your users.json (same as auth_local.py)
USERS_FILE = os.path.join(os.path.dirname(__file__), "users.json")

def hash_password(password: str) -> str:
    """Hash a password using bcrypt"""
    return bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()

def save_users(users: dict):
    with open(USERS_FILE, "w") as f:
        json.dump(users, f, indent=2)

if __name__ == "__main__":
    # Define your test users here
    test_users = {
        "admin": "admin123",
        "test": "test123",
    }

    users_hashed = {}
    for username, password in test_users.items():
        users_hashed[username] = {"password": hash_password(password)}

    save_users(users_hashed)
    print(f"✅ Created {len(users_hashed)} users in {USERS_FILE}")
