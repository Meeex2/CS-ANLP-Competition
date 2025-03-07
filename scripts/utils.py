import bisect
import functools
import re

import pandas as pd


def to_codepoint(s):
    if isinstance(s, str) and s.startswith("\\u"):
        return int(s[2:], 16)
    elif isinstance(s, str) and s.startswith("U+"):
        return int(s[2:], 16)
    else:
        return int(s)


@functools.lru_cache(maxsize=1)
def load_script_ranges():
    df = pd.read_csv("output/unicode_ranges.csv")
    df["start"] = df["range_start"].apply(to_codepoint)
    df["end"] = df["range_end"].apply(to_codepoint)
    # Sort ranges by the start value.
    ranges = sorted(df.itertuples(index=False), key=lambda r: r.start)
    # Cache the unique language group names from the CSV.
    lang_names = df["language_group_name"].unique().tolist()
    # Also return a list of all start values for binary search.
    range_starts = [r.start for r in ranges]
    return ranges, range_starts, lang_names


def detect_script(text):
    ranges, range_starts, lang_names = load_script_ranges()

    # Initialize counts for each language group and "Unknown".
    script_counts = {lang: 0 for lang in lang_names}
    script_counts["Unknown"] = 0

    # Clean text (removing leading/trailing spaces and internal spaces).
    text = text.strip().replace(" ", "")
    for char in text:
        code = ord(char)
        # Locate the rightmost range whose start is <= code.
        idx = bisect.bisect_right(range_starts, code) - 1
        if idx >= 0:
            r = ranges[idx]
            if r.start <= code <= r.end:
                script_counts[r.language_group_name] += 1
                continue  # Skip the "Unknown" count.
        script_counts["Unknown"] += 1

    return script_counts


def filter_majority_script(text):
    """
    Keeps only the characters in the majority script so that text is uniform in its script.
    """
    script_counts = detect_script(text)
    # Find the majority script (excluding "Unknown").
    majority_script = max(
        (script for script in script_counts if script != "Unknown"),
        key=script_counts.get,
    )

    # Filter text to keep only characters in the majority script.
    ranges, range_starts, _ = load_script_ranges()
    filtered_text = []
    for char in text:
        code = ord(char)
        if char.isspace() or char in "()[]{}":
            filtered_text.append(char)
            continue
        idx = bisect.bisect_right(range_starts, code) - 1
        if idx >= 0:
            r = ranges[idx]
            if r.start <= code <= r.end and r.language_group_name == majority_script:
                filtered_text.append(char)

    return "".join(filtered_text)


def remove_links_and_tags(text: str):
    """
    Removes internet links, tags that begin with @, and hashtags.
    """
    # Remove URLs (e.g., starting with http://, https://, or www.)
    text = re.sub(r"http\S+|www\.\S+", "", text)
    # Remove @tags (words that begin with @ followed by alphanumeric or underscore characters)
    text = re.sub(r"@\w+", "", text)
    # Remove hashtags (words that begin with # followed by alphanumeric or underscore characters)
    text = re.sub(r"#\w+", "", text)
    # Remove multiple spaces between words.
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def remove_emojis(
    text_series: pd.Series, csv_path: str = "data/emoji_unicodes.csv"
) -> pd.Series:
    """
    Removes emojis from a pandas Series of text using a CSV file containing emoji unicode codes.

    Args:
        text_series (pd.Series): Series of text containing emojis
        csv_path (str): Path to CSV file with emoji unicode codes

    Returns:
        pd.Series: Series with emojis removed from each text entry
    """
    # Read emoji codes from CSV
    emoji_df = pd.read_csv(csv_path)

    # Convert codes to actual unicode characters, handling space-separated codes
    emoji_chars = []
    for code in emoji_df["code"]:
        # Split code if it contains multiple space-separated values
        parts = code.split()
        try:
            # Convert each part to unicode character and combine
            chars = "".join(chr(int(part, 16)) for part in parts)
            emoji_chars.append(chars)
        except ValueError:
            continue  # Skip invalid codes

    # Create regex pattern for all emoji characters
    emoji_pattern = "|".join(map(re.escape, emoji_chars))

    # Function to remove emojis from a single text entry
    def remove_emojis_from_text(text):
        clean_text = re.sub(emoji_pattern, "", text)
        clean_text = re.sub(r"\s+", " ", clean_text)
        return clean_text.strip()

    # Apply the function to the entire Series
    return text_series.apply(remove_emojis_from_text)
