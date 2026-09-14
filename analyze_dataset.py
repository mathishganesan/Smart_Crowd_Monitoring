from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import skew, pearsonr
import matplotlib.pyplot as plt


# ============================================================
# CONFIGURATION
# ============================================================

# Project root = location of this Python file
PROJECT_ROOT = Path(__file__).resolve().parent

# Dataset directory
DATASET_DIR = PROJECT_ROOT / "dataset"

# Dataset splits
SPLITS = ["train", "valid", "test"]

# Output directory
OUTPUT_DIR = PROJECT_ROOT / "dataset_analysis"
OUTPUT_DIR.mkdir(exist_ok=True)


# ============================================================
# FUNCTION: READ YOLO LABEL FILES
# ============================================================

def analyze_split(split_name):
    """
    Read all YOLO label files for a dataset split.

    YOLO format:
    class_id x_center y_center width height
    """

    labels_dir = DATASET_DIR / split_name / "labels"

    if not labels_dir.exists():
        print(f"[WARNING] Labels directory not found: {labels_dir}")
        return []

    records = []

    label_files = sorted(labels_dir.glob("*.txt"))

    print(f"\nAnalyzing {split_name.upper()} dataset...")
    print(f"Label files found: {len(label_files)}")

    for label_file in label_files:

        try:
            with open(label_file, "r", encoding="utf-8") as file:
                lines = [
                    line.strip()
                    for line in file
                    if line.strip()
                ]

        except Exception as e:
            print(f"[ERROR] Could not read {label_file}: {e}")
            continue

        head_count = 0
        widths = []
        heights = []

        for line in lines:

            parts = line.split()

            # Expected YOLO format:
            # class x_center y_center width height
            if len(parts) != 5:
                continue

            try:
                class_id = int(parts[0])
                x_center = float(parts[1])
                y_center = float(parts[2])
                width = float(parts[3])
                height = float(parts[4])

            except ValueError:
                continue

            # Current project has one class:
            # 0 = head
            if class_id == 0:

                head_count += 1

                widths.append(width)
                heights.append(height)

        # Store image-level statistics
        records.append({
            "split": split_name,
            "image_name": label_file.stem,
            "head_count": head_count,
            "average_head_width": (
                np.mean(widths) if widths else 0
            ),
            "average_head_height": (
                np.mean(heights) if heights else 0
            ),
            "average_head_area": (
                np.mean(
                    [w * h for w, h in zip(widths, heights)]
                )
                if widths
                else 0
            )
        })

    return records


# ============================================================
# READ ALL DATASET SPLITS
# ============================================================

all_records = []

for split in SPLITS:
    records = analyze_split(split)
    all_records.extend(records)


if not all_records:
    print("\nERROR: No label files were found.")
    print("Check your dataset structure:")
    print("dataset/train/labels/")
    print("dataset/valid/labels/")
    print("dataset/test/labels/")
    raise SystemExit


df = pd.DataFrame(all_records)


# ============================================================
# BASIC DATASET INFORMATION
# ============================================================

total_images = len(df)
total_heads = int(df["head_count"].sum())

mean_heads = df["head_count"].mean()
median_heads = df["head_count"].median()

minimum_heads = df["head_count"].min()
maximum_heads = df["head_count"].max()

std_heads = df["head_count"].std()
variance_heads = df["head_count"].var()

q1 = df["head_count"].quantile(0.25)
q3 = df["head_count"].quantile(0.75)

iqr = q3 - q1


# ============================================================
# SKEWNESS
# ============================================================

if df["head_count"].nunique() > 1:
    skewness_value = skew(
        df["head_count"],
        bias=False
    )
else:
    skewness_value = 0


# ============================================================
# OUTLIER DETECTION USING IQR
# ============================================================

lower_bound = q1 - (1.5 * iqr)
upper_bound = q3 + (1.5 * iqr)

outliers = df[
    (df["head_count"] < lower_bound)
    |
    (df["head_count"] > upper_bound)
]

number_of_outliers = len(outliers)


# ============================================================
# CORRELATION
# ============================================================

# Correlation between:
# Head count and average head area

valid_corr_df = df[
    (df["head_count"] > 0)
    &
    (df["average_head_area"] > 0)
]

if (
    len(valid_corr_df) >= 2
    and valid_corr_df["head_count"].nunique() > 1
    and valid_corr_df["average_head_area"].nunique() > 1
):

    correlation, correlation_pvalue = pearsonr(
        valid_corr_df["head_count"],
        valid_corr_df["average_head_area"]
    )

else:
    correlation = np.nan
    correlation_pvalue = np.nan


# ============================================================
# PRINT REPORT
# ============================================================

print("\n")
print("=" * 65)
print("        SCUT-HEAD DATASET STATISTICAL ANALYSIS")
print("=" * 65)

print(f"\nTotal images analyzed       : {total_images:,}")
print(f"Total annotated heads      : {total_heads:,}")

print("\n--- CENTER OF DATA ---")
print(f"Mean heads/image            : {mean_heads:.2f}")
print(f"Median heads/image          : {median_heads:.2f}")

print("\n--- SPREAD OF DATA ---")
print(f"Minimum heads/image         : {minimum_heads}")
print(f"Maximum heads/image         : {maximum_heads}")
print(f"Standard deviation          : {std_heads:.2f}")
print(f"Variance                    : {variance_heads:.2f}")
print(f"Q1                          : {q1:.2f}")
print(f"Q3                          : {q3:.2f}")
print(f"IQR                         : {iqr:.2f}")

print("\n--- SKEWNESS ---")
print(f"Skewness                    : {skewness_value:.4f}")

if skewness_value > 0:
    print("Interpretation              : Right / Positive skew")
elif skewness_value < 0:
    print("Interpretation              : Left / Negative skew")
else:
    print("Interpretation              : Approximately symmetric")

print("\n--- OUTLIERS ---")
print(f"Lower IQR boundary          : {lower_bound:.2f}")
print(f"Upper IQR boundary          : {upper_bound:.2f}")
print(f"Number of outlier images    : {number_of_outliers}")

print("\n--- CORRELATION ---")

if not np.isnan(correlation):
    print(
        "Head count vs average head area"
    )
    print(f"Pearson correlation (r)     : {correlation:.4f}")
    print(f"P-value                     : {correlation_pvalue:.6f}")

else:
    print("Correlation could not be calculated.")

print("\n--- SPLIT INFORMATION ---")

for split in SPLITS:

    split_df = df[df["split"] == split]

    if len(split_df) == 0:
        continue

    print(f"\n{split.upper()}")
    print(f"Images                      : {len(split_df):,}")
    print(f"Heads                       : {split_df['head_count'].sum():,}")
    print(
        f"Mean heads/image            : "
        f"{split_df['head_count'].mean():.2f}"
    )
    print(
        f"Minimum heads/image         : "
        f"{split_df['head_count'].min()}"
    )
    print(
        f"Maximum heads/image         : "
        f"{split_df['head_count'].max()}"
    )


# ============================================================
# SAVE IMAGE-LEVEL DATA
# ============================================================

csv_path = OUTPUT_DIR / "image_level_statistics.csv"

df.to_csv(
    csv_path,
    index=False
)

print("\n")
print(f"Image-level statistics saved to:")
print(csv_path)


# ============================================================
# SAVE OUTLIER INFORMATION
# ============================================================

outlier_path = OUTPUT_DIR / "outliers.csv"

outliers.to_csv(
    outlier_path,
    index=False
)

print(f"Outlier information saved to:")
print(outlier_path)


# ============================================================
# SAVE TEXT REPORT
# ============================================================

report_path = OUTPUT_DIR / "dataset_report.txt"

with open(
    report_path,
    "w",
    encoding="utf-8"
) as report:

    report.write(
        "SCUT-HEAD DATASET STATISTICAL ANALYSIS\n"
    )
    report.write("=" * 60 + "\n\n")

    report.write(
        f"Total images analyzed : {total_images:,}\n"
    )

    report.write(
        f"Total annotated heads : {total_heads:,}\n\n"
    )

    report.write("CENTER OF DATA\n")
    report.write("-" * 30 + "\n")
    report.write(
        f"Mean heads/image      : {mean_heads:.2f}\n"
    )
    report.write(
        f"Median heads/image    : {median_heads:.2f}\n\n"
    )

    report.write("SPREAD\n")
    report.write("-" * 30 + "\n")
    report.write(
        f"Minimum               : {minimum_heads}\n"
    )
    report.write(
        f"Maximum               : {maximum_heads}\n"
    )
    report.write(
        f"Standard deviation    : {std_heads:.2f}\n"
    )
    report.write(
        f"Variance              : {variance_heads:.2f}\n"
    )
    report.write(
        f"Q1                    : {q1:.2f}\n"
    )
    report.write(
        f"Q3                    : {q3:.2f}\n"
    )
    report.write(
        f"IQR                   : {iqr:.2f}\n\n"
    )

    report.write("SKEWNESS\n")
    report.write("-" * 30 + "\n")
    report.write(
        f"Skewness              : "
        f"{skewness_value:.4f}\n\n"
    )

    report.write("OUTLIERS\n")
    report.write("-" * 30 + "\n")
    report.write(
        f"Lower boundary       : "
        f"{lower_bound:.2f}\n"
    )
    report.write(
        f"Upper boundary       : "
        f"{upper_bound:.2f}\n"
    )
    report.write(
        f"Outlier images       : "
        f"{number_of_outliers}\n\n"
    )

    report.write("CORRELATION\n")
    report.write("-" * 30 + "\n")

    if not np.isnan(correlation):

        report.write(
            "Head count vs average head area\n"
        )

        report.write(
            f"Pearson r            : "
            f"{correlation:.4f}\n"
        )

        report.write(
            f"P-value              : "
            f"{correlation_pvalue:.6f}\n"
        )

    else:

        report.write(
            "Correlation could not be calculated.\n"
        )


print(f"\nFull report saved to:")
print(report_path)


# ============================================================
# GRAPH 1: HEAD COUNT DISTRIBUTION
# ============================================================

plt.figure(figsize=(10, 6))

plt.hist(
    df["head_count"],
    bins=30,
    edgecolor="black"
)

plt.xlabel("Number of Heads per Image")
plt.ylabel("Number of Images")
plt.title("SCUT-HEAD: Head Count Distribution")
plt.grid(axis="y", alpha=0.3)

plt.tight_layout()

histogram_path = OUTPUT_DIR / "head_count_distribution.png"

plt.savefig(
    histogram_path,
    dpi=300
)

plt.close()


# ============================================================
# GRAPH 2: BOX PLOT / OUTLIERS
# ============================================================

plt.figure(figsize=(8, 6))

plt.boxplot(
    df["head_count"],
    vert=True
)

plt.ylabel("Number of Heads per Image")
plt.title("SCUT-HEAD: Head Count Box Plot")
plt.grid(axis="y", alpha=0.3)

plt.tight_layout()

boxplot_path = OUTPUT_DIR / "head_count_boxplot.png"

plt.savefig(
    boxplot_path,
    dpi=300
)

plt.close()


# ============================================================
# GRAPH 3: PROBABILITY DISTRIBUTION
# ============================================================

plt.figure(figsize=(10, 6))

counts = df["head_count"].value_counts().sort_index()

probabilities = counts / total_images

plt.plot(
    probabilities.index,
    probabilities.values,
    marker="o"
)

plt.xlabel("Number of Heads per Image")
plt.ylabel("Probability")
plt.title("SCUT-HEAD: Probability Distribution of Head Counts")
plt.grid(True, alpha=0.3)

plt.tight_layout()

probability_path = (
    OUTPUT_DIR / "head_count_probability_distribution.png"
)

plt.savefig(
    probability_path,
    dpi=300
)

plt.close()


# ============================================================
# GRAPH 4: HEAD COUNT VS AVERAGE HEAD AREA
# ============================================================

if len(valid_corr_df) >= 2:

    plt.figure(figsize=(10, 6))

    plt.scatter(
        valid_corr_df["head_count"],
        valid_corr_df["average_head_area"]
    )

    plt.xlabel("Number of Heads per Image")
    plt.ylabel("Average Head Bounding-Box Area")
    plt.title(
        "SCUT-HEAD: Head Count vs Average Head Area"
    )

    plt.grid(True, alpha=0.3)

    plt.tight_layout()

    correlation_path = (
        OUTPUT_DIR / "head_count_vs_head_area.png"
    )

    plt.savefig(
        correlation_path,
        dpi=300
    )

    plt.close()


# ============================================================
# COMPLETION MESSAGE
# ============================================================

print("\n")
print("=" * 65)
print("                 ANALYSIS COMPLETED")
print("=" * 65)

print("\nGenerated files:")

print("1. image_level_statistics.csv")
print("2. outliers.csv")
print("3. dataset_report.txt")
print("4. head_count_distribution.png")
print("5. head_count_boxplot.png")
print("6. head_count_probability_distribution.png")
print("7. head_count_vs_head_area.png")

print("\nAll files are inside:")
print(OUTPUT_DIR)

print("\nDone.")