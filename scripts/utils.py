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

    max_lang = max(script_counts, key=script_counts.get)
    return max_lang


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
