# data_prep.py
"""
Read the original survey data, clean it, and build datasets for analysis.

Two layers:
1) prepare_base_dataset(): basic cleaning + AI usage intensity + log_salary
2) prepare_model_dataset(): add YearsCodeNum / ExpGroup / CountryGroup
                            on top of the base dataset.

Usage:
    python data_prep.py
"""

import os
import numpy as np
import pandas as pd
from config import DATA_DIR, PUBLIC_CSV, SCHEMA_CSV

# We currently use only these 18 raw fields; more can be added later if needed.
RAW_COLUMNS = [
    "ResponseId",
    "Age",
    "EdLevel",
    "Employment",
    "WorkExp",
    "YearsCode",
    "DevType",
    "OrgSize",
    "RemoteWork",
    "Industry",
    "Country",
    "AISelect",
    "AIToolCurrently partially AI",
    "AIToolDon't plan to use AI for this task",
    "AIToolPlan to partially use AI",
    "AIToolPlan to mostly use AI",
    "AIToolCurrently mostly AI",
    "ConvertedCompYearly",
]

# All AITool-related columns (used to count total tasks)
AI_COLUMNS = [
    "AIToolCurrently partially AI",
    "AIToolDon't plan to use AI for this task",
    "AIToolPlan to partially use AI",
    "AIToolPlan to mostly use AI",
    "AIToolCurrently mostly AI",
]


def load_raw_data():
    """Read the main survey data and the schema from data/ folder."""
    public_path = os.path.join(DATA_DIR, PUBLIC_CSV)
    schema_path = os.path.join(DATA_DIR, SCHEMA_CSV)

    df_raw = pd.read_csv(public_path)
    df_schema = pd.read_csv(schema_path)

    return df_raw, df_schema


def _parse_tasks(cell):
    """Split a multi-select string into a set of tasks; return empty set for NaN."""
    if pd.isna(cell):
        return set()
    tasks = set()
    for t in str(cell).split(";"):
        t = t.strip()
        if t:
            tasks.add(t)
    return tasks


def _compute_ai_usage_row(row):
    """
    Compute AI usage intensity variables for a single row:

    - AI_TotalTasks: number of distinct tasks appearing in any AITool column
    - AI_NumMostly: number of tasks currently "mostly AI"
    - AI_NumPartial: number of tasks currently "partially AI"
    - AI_MostlyShare = NumMostly / TotalTasks
    - AI_PartialShare = NumPartial / TotalTasks
    - AI_Index = (NumMostly + 0.5 * NumPartial) / TotalTasks
    """
    # All distinct tasks that appear in any AITool column
    all_tasks = set()
    for col in AI_COLUMNS:
        all_tasks |= _parse_tasks(row.get(col, np.nan))

    total_tasks = len(all_tasks)

    mostly = _parse_tasks(row.get("AIToolCurrently mostly AI", np.nan))
    partial = _parse_tasks(row.get("AIToolCurrently partially AI", np.nan))

    num_mostly = len(mostly)
    num_partial = len(partial)

    if total_tasks == 0:
        return pd.Series(
            {
                "AI_TotalTasks": 0,
                "AI_NumMostly": num_mostly,
                "AI_NumPartial": num_partial,
                "AI_MostlyShare": np.nan,
                "AI_PartialShare": np.nan,
                "AI_Index": np.nan,
            }
        )

    return pd.Series(
        {
            "AI_TotalTasks": total_tasks,
            "AI_NumMostly": num_mostly,
            "AI_NumPartial": num_partial,
            "AI_MostlyShare": float(num_mostly) / total_tasks,
            "AI_PartialShare": float(num_partial) / total_tasks,
            "AI_Index": (num_mostly + 0.5 * num_partial) / total_tasks,
        }
    )


def prepare_base_dataset():
    """
    1) Read raw data
    2) Select needed columns
    3) Drop respondents with missing / non-positive salary or zero AI tasks
    4) Construct AI usage intensity variables
    5) Winsorize salary at 1% and 99% and create log_salary

    Returns:
        Cleaned base DataFrame.
    """
    df_raw, df_schema = load_raw_data()

    # Keep only the columns we need
    df = df_raw[RAW_COLUMNS].copy()

    # Convert salary to numeric and drop <= 0 or missing values
    df["ConvertedCompYearly"] = pd.to_numeric(
        df["ConvertedCompYearly"], errors="coerce"
    )
    df = df[df["ConvertedCompYearly"] > 0]

    # Compute AI usage intensity variables
    ai_usage = df.apply(_compute_ai_usage_row, axis=1)
    df = pd.concat([df, ai_usage], axis=1)

    # Drop respondents who have zero AI-related tasks
    df = df[df["AI_TotalTasks"] > 0]

    # Winsorize salary at 1% / 99% quantiles
    q_low, q_high = df["ConvertedCompYearly"].quantile([0.01, 0.99])
    df = df[
        (df["ConvertedCompYearly"] >= q_low)
        & (df["ConvertedCompYearly"] <= q_high)
    ].copy()

    # Log salary (for regressions)
    df["log_salary"] = np.log(df["ConvertedCompYearly"])

    return df


# ===== Step 2: experience group + country group =====

def _years_to_float(val):
    """Convert YearsCode string to numeric, handling 'Less than 1 year' etc."""
    if pd.isna(val):
        return np.nan
    s = str(val).strip()
    if s.lower().startswith("less than"):
        return 0.5
    if s.lower().startswith("more than"):
        # e.g. "More than 50 years"
        for token in s.split():
            try:
                return float(token)
            except ValueError:
                continue
        return 50.0
    try:
        return float(s)
    except ValueError:
        return np.nan


def _exp_group_from_years(years):
    """Map numeric years of coding to experience groups: low / mid / high."""
    if pd.isna(years):
        return np.nan
    if years < 4:
        return "low"
    elif years < 10:
        return "mid"
    else:
        return "high"


def _add_grouping_variables(df):
    """
    Add grouping variables on top of the base dataset:

    - YearsCodeNum: numeric years of coding
    - ExpGroup: experience group (low / mid / high)
    - CountryGroup: top 10 countries kept separate, all others merged to 'Other'
    """
    df = df.copy()

    # YearsCode → numeric
    df["YearsCodeNum"] = df["YearsCode"].apply(_years_to_float)
    df["ExpGroup"] = df["YearsCodeNum"].apply(_exp_group_from_years)

    # Country → CountryGroup (top 10 kept, the rest merged to 'Other')
    counts = df["Country"].value_counts()
    top_countries = list(counts.head(10).index)

    def map_country(c):
        if pd.isna(c):
            return "Other"
        if c in top_countries:
            return c
        return "Other"

    df["CountryGroup"] = df["Country"].apply(map_country)

    return df


def prepare_model_dataset():
    """
    Build the final dataset used for modelling and plotting:
    base cleaning + experience group + country group.
    """
    df = prepare_base_dataset()
    df = _add_grouping_variables(df)
    return df


if __name__ == "__main__":
    df_model = prepare_model_dataset()
    print("Shape of modelling dataset:", df_model.shape)
    print()
    print(
        df_model[
            [
                "ConvertedCompYearly",
                "log_salary",
                "AI_TotalTasks",
                "AI_MostlyShare",
                "AI_PartialShare",
                "AI_Index",
                "YearsCode",
                "YearsCodeNum",
                "ExpGroup",
                "Country",
                "CountryGroup",
            ]
        ].head()
    )
