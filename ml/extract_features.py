import pandas as pd
import re
import math
from urllib.parse import urlparse


# ==============================
# Suspicious words
# ==============================

SUSPICIOUS_WORDS = [
    "login",
    "signin",
    "verify",
    "verification",
    "account",
    "password",
    "passwd",
    "bank",
    "secure",
    "update",
    "confirm",
    "authenticate",
    "wallet",
    "payment",
    "billing",
    "credential"
]


# ==============================
# Entropy calculation
# ==============================

def calculate_entropy(text):
    if not text:
        return 0

    frequency = {}

    for char in text:
        frequency[char] = frequency.get(char, 0) + 1

    entropy = 0

    for count in frequency.values():
        probability = count / len(text)
        entropy -= probability * math.log2(probability)

    return entropy


# ==============================
# Feature extraction
# ==============================

def extract_features(url):

    parsed = urlparse(url)

    # URL length
    url_length = len(url)

    # Number of dots
    num_dots = url.count(".")

    # HTTPS
    has_https = 1 if parsed.scheme.lower() == "https" else 0

    # Check whether hostname is an IP address
    hostname = parsed.hostname or ""

    ip_pattern = r"^(?:\d{1,3}\.){3}\d{1,3}$"
    has_ip = 1 if re.match(ip_pattern, hostname) else 0

    # Number of subdirectories
    path = parsed.path

    if path:
        num_subdirs = len(
            [part for part in path.split("/") if part]
        )
    else:
        num_subdirs = 0

    # Number of query parameters
    if parsed.query:
        num_params = len(parsed.query.split("&"))
    else:
        num_params = 0

    # Suspicious words
    url_lower = url.lower()

    suspicious_words = sum(
        1 for word in SUSPICIOUS_WORDS
        if word in url_lower
    )

    # Special characters
    special_characters = r"[@?=&%_\-:#]"

    special_char_count = len(
        re.findall(special_characters, url)
    )

    # Digits
    digits_count = sum(
        char.isdigit()
        for char in url
    )

    # Entropy
    entropy = calculate_entropy(url)

    return {
        "url_length": url_length,
        "num_dots": num_dots,
        "has_https": has_https,
        "has_ip": has_ip,
        "num_subdirs": num_subdirs,
        "num_params": num_params,
        "suspicious_words": suspicious_words,
        "special_char_count": special_char_count,
        "digits_count": digits_count,
        "entropy": entropy
    }


# ==============================
# Load dataset
# ==============================

df = pd.read_csv("phishing_url_dataset_unique.csv")


# ==============================
# Extract features
# ==============================

features = df["url"].apply(extract_features)

features_df = pd.DataFrame(features.tolist())


# Add label
features_df["label"] = df["label"].values


# ==============================
# Save feature dataset
# ==============================

features_df.to_csv(
    "phishing_features_new.csv",
    index=False
)


print("Feature extraction completed!")

print("\nDataset shape:")
print(features_df.shape)

print("\nColumns:")
print(features_df.columns.tolist())

print("\nFirst 5 rows:")
print(features_df.head())

print("\nLabel distribution:")
print(features_df["label"].value_counts())

print("\nSaved as:")
print("phishing_features_new.csv")