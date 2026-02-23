import os
import re
from datetime import datetime

def slugify(text):
    text = text.lower().strip()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    return text.strip("_")


def new_history_filename(folder, title=None):
    date = datetime.now().strftime("%Y-%m-%d")
    os.makedirs(folder, exist_ok=True)

    # All files for today
    existing = [f for f in os.listdir(folder) if f.startswith(date)]

    # Auto-number chat sessions
    count = sum(1 for f in existing if "_chat_" in f) + 1

    # If user typed a title
    if title:
        title = slugify(title)
        return f"{date}_{title}.json"

    # Default numbered session
    return f"{date}_chat_{count:02d}.json"
