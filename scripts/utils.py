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


def remove_emojis(text: str):
    """
    Removes emojis from the text.
    """
    emoji_pattern = re.compile(
        "["
        "\U0001f600-\U0001f64f"  # emoticons
        "\U0001f300-\U0001f5ff"  # symbols & pictographs
        "\U0001f680-\U0001f6ff"  # transport & map symbols
        "\U0001f1e0-\U0001f1ff"  # flags (iOS)
        "\U00002702-\U000027b0"
        "\U000024c2-\U0001f251"
        "]+",
        flags=re.UNICODE,
    )
    return emoji_pattern.sub(r"", text)
