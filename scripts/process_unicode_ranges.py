import csv
import re

# Input and output file paths
input_csv = "data/unicode_ranges.csv"
output_csv = "output/unicode_ranges.csv"

# Regex to extract hexadecimal values from the Block range
hex_pattern = re.compile(r"U\+([0-9A-Fa-f]+)")

# Open the input CSV file and create the output CSV file
with (
    open(input_csv, mode="r", newline="", encoding="utf-8") as infile,
    open(output_csv, mode="w", newline="", encoding="utf-8") as outfile,
):
    reader = csv.DictReader(infile)
    fieldnames = ["language_group_name", "range_start", "range_end"]
    writer = csv.DictWriter(outfile, fieldnames=fieldnames)

    # Write the new header
    writer.writeheader()

    # Process each row
    for row in reader:
        # Rename the "Group name" column to "language_group_name"
        language_group_name = row["Group name"]

        # Extract the start and end values from the "Block range" column
        block_range = row["Block range"]
        matches = hex_pattern.findall(block_range)

        if len(matches) == 2:
            range_start = f"\\u{int(matches[0], 16):04x}"
            range_end = f"\\u{int(matches[1], 16):04x}"
        else:
            range_start, range_end = "", ""  # Handle invalid formats

        # Write the modified row to the output CSV
        writer.writerow(
            {
                "language_group_name": language_group_name,
                "range_start": range_start,
                "range_end": range_end,
            }
        )

print(f"Modified CSV saved to {output_csv}")
