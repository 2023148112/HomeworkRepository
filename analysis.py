"""
Analysis script for Stack Overflow developer survey data.

We use the cleaned modelling dataset from data_prep.py and run 5 parts:

Part 1: AI usage intensity vs salary
    - Distribution of AI_Index
    - Median salary by AI_Index quartile (overall)
    - Median salary by AI_Index quartile within experience groups (low/mid/high)
    - Main OLS regression: log_salary ~ AI_Index + controls
    - Decomposed regression: log_salary ~ AI_MostlyShare + AI_PartialShare + controls

Part 2: Who uses AI?
    - Mean AI_Index / AI_TotalTasks by experience group
    - Mean AI_Index / AI_TotalTasks by country group (USA / UK / Other)
    - Mean AI_Index / AI_TotalTasks by age group

Part 3: Age and salary
    - Median salary by age group (overall)
    - Median salary by age group within country (USA / UK / Other)

Part 4: Role, age and salary
    - For the top 8 primary roles, show median age (approx) and median salary

Part 5: Stepwise effects of AI use
    - Classify respondents into three AI use groups:
      (no use / partial_only / mostly)
    - Compare median yearly salary across the three groups
      to see whether partial_only lies between no use and mostly
    - Run a regression log_salary ~ AI_UseGroup + controls
      to test the stepwise relationship between AI use intensity and pay
"""

import os
from typing import Dict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf

from data_prep import prepare_model_dataset
from config import OUTPUT_DIR


# Utility

def ensure_output_dir() -> None:
    """Create output directory if it does not exist."""
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)


# Part 1: AI usage intensity vs salary

def part1_ai_vs_salary(df: pd.DataFrame) -> None:
    """
    Part 1:
    - Distribution of AI_Index
    - Median salary by AI_Index quartile (overall)
    - Median salary by AI_Index quartile within experience groups
    - Main regression: log_salary ~ AI_Index + controls
    - Decomposed regression: log_salary ~ AI_MostlyShare + AI_PartialShare + controls
    """
    ensure_output_dir()

    # Keep only rows with AI_Index
    df_ai = df[df["AI_Index"].notna()].copy()

    # 1) Distribution of AI_Index
    plt.figure()
    df_ai["AI_Index"].hist(bins=20)
    plt.xlabel("AI_Index (overall AI usage intensity)")
    plt.ylabel("Count")
    plt.title("Distribution of AI_Index")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "part1_ai_index_hist.png"))
    plt.close()

    # 2) AI_Index quartiles (allow fewer than 4 bins if there are duplicate edges)
    valid = df_ai["AI_Index"].notna()
    quartile = pd.qcut(
        df_ai.loc[valid, "AI_Index"],
        4,
        duplicates="drop",
    )
    n_bins = quartile.cat.categories.size
    labels = [f"Q{i + 1}" for i in range(n_bins)]
    quartile = quartile.cat.rename_categories(labels)
    df_ai.loc[valid, "AI_Index_quartile"] = quartile

    # Median salary by AI quartile (overall)
    salary_by_q = (
        df_ai.groupby("AI_Index_quartile")["ConvertedCompYearly"]
        .median()
        .reset_index()
        .sort_values("AI_Index_quartile")
    )
    print("=== Part 1 (overall): median salary by AI_Index quartile ===")
    print(salary_by_q)
    salary_by_q.to_csv(
        os.path.join(OUTPUT_DIR, "part1_salary_by_ai_quartile.csv"), index=False
    )

    plt.figure()
    plt.bar(
        salary_by_q["AI_Index_quartile"].astype(str),
        salary_by_q["ConvertedCompYearly"],
    )
    plt.xlabel("AI_Index quartile (from low to high)")
    plt.ylabel("Median yearly compensation")
    plt.title("Median salary by AI usage quartile (overall)")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "part1_salary_by_ai_quartile.png"))
    plt.close()

    # 3) Median salary by AI quartile within experience groups
    salary_by_q_exp = (
        df_ai.groupby(["ExpGroup", "AI_Index_quartile"])["ConvertedCompYearly"]
        .median()
        .unstack("AI_Index_quartile")
        .reindex(["low", "mid", "high"])
    )
    print("\n=== Part 1 (by experience): median salary by AI_Index quartile ===")
    print(salary_by_q_exp)
    salary_by_q_exp.to_csv(
        os.path.join(OUTPUT_DIR, "part1_salary_by_ai_quartile_expgroup.csv")
    )

    label_exp: Dict[str, str] = {
        "low": "Low experience (<4 years)",
        "mid": "Mid experience (4–9 years)",
        "high": "High experience (10+ years)",
    }

    for exp in ["low", "mid", "high"]:
        if exp not in salary_by_q_exp.index:
            continue
        plt.figure()
        row = salary_by_q_exp.loc[exp].dropna()
        row.plot(kind="bar", rot=0)
        plt.xlabel("AI_Index quartile (from low to high)")
        plt.ylabel("Median yearly compensation")
        plt.title(
            f"Median salary by AI usage quartile\n{label_exp.get(exp, exp)}"
        )
        plt.tight_layout()
        fname = f"part1_salary_by_ai_quartile_exp_{exp}.png"
        plt.savefig(os.path.join(OUTPUT_DIR, fname))
        plt.close()

    # 4) Main regression: log_salary ~ AI_Index + controls
    df_reg = df.copy()
    df_reg = df_reg[
        df_reg["AI_Index"].notna()
        & df_reg["log_salary"].notna()
        & df_reg["ExpGroup"].notna()
        & df_reg["CountryGroup"].notna()
    ]

    formula = (
        "log_salary ~ AI_Index"
        " + C(ExpGroup)"
        " + C(CountryGroup)"
        " + C(OrgSize)"
        " + C(RemoteWork)"
        " + C(EdLevel)"
        " + C(Age)"
    )

    model = smf.ols(formula=formula, data=df_reg).fit()
    print("\n=== Part 1 (main regression): log_salary ~ AI_Index + controls ===")
    print(model.summary())

    coef_table = model.summary2().tables[1]
    coef_table.to_csv(
        os.path.join(OUTPUT_DIR, "part1_main_regression_coefs.csv")
    )

    beta_ai = model.params.get("AI_Index", np.nan)
    if not np.isnan(beta_ai):
        pct_change = (np.exp(beta_ai * 0.1) - 1.0) * 100
        print(
            f"\nInterpretation: after controlling for experience, country, "
            f"org size, remote work, education and age, a 0.1 increase in "
            f"AI_Index is associated with about {pct_change:.2f}% change in "
            f"predicted yearly salary (approximate)."
        )

    # 5) Decomposed regression: Mostly vs Partial AI usage
    df_dec = df.copy()
    df_dec = df_dec[
        df_dec["log_salary"].notna()
        & df_dec["AI_MostlyShare"].notna()
        & df_dec["AI_PartialShare"].notna()
        & df_dec["ExpGroup"].notna()
        & df_dec["CountryGroup"].notna()
    ]

    formula2 = (
        "log_salary ~ AI_MostlyShare + AI_PartialShare"
        " + C(ExpGroup)"
        " + C(CountryGroup)"
        " + C(OrgSize)"
        " + C(RemoteWork)"
        " + C(EdLevel)"
        " + C(Age)"
    )

    model2 = smf.ols(formula=formula2, data=df_dec).fit()
    print(
        "\n=== Part 1 (decomposed regression): "
        "log_salary ~ AI_MostlyShare + AI_PartialShare + controls ==="
    )
    print(model2.summary())

    coef_table2 = model2.summary2().tables[1]
    coef_table2.to_csv(
        os.path.join(OUTPUT_DIR, "part1_decomposed_regression_coefs.csv")
    )

    b_most = model2.params.get("AI_MostlyShare", np.nan)
    b_part = model2.params.get("AI_PartialShare", np.nan)
    if not np.isnan(b_most):
        pct = (np.exp(b_most * 0.1) - 1.0) * 100
        print(
            f"\nInterpretation: a 0.1 increase in AI_MostlyShare "
            f"is associated with about {pct:.2f}% change in salary "
            f"(approximate)."
        )
    if not np.isnan(b_part):
        pct = (np.exp(b_part * 0.1) - 1.0) * 100
        print(
            f"Interpretation: a 0.1 increase in AI_PartialShare "
            f"is associated with about {pct:.2f}% change in salary "
            f"(approximate)."
        )



# Part 2: Who uses AI? (experience / country / age)

def part2_ai_usage_profile(df: pd.DataFrame) -> None:
    """
    Part 2:
    - Mean AI_Index / AI_TotalTasks by experience group
    - Mean AI_Index / AI_TotalTasks by country (USA / UK / Other)
    - Mean AI_Index / AI_TotalTasks by age group
    """
    ensure_output_dir()

    df_ai = df[df["AI_Index"].notna()].copy()

    # 1) By experience group
    usage_by_exp = df_ai.groupby("ExpGroup")[["AI_Index", "AI_TotalTasks"]].mean()
    print("\n=== Part 2: mean AI usage by experience group ===")
    print(usage_by_exp)
    usage_by_exp.to_csv(
        os.path.join(OUTPUT_DIR, "part2_ai_usage_by_exp.csv")
    )

    plt.figure()
    usage_by_exp["AI_Index"].reindex(["low", "mid", "high"]).plot(
        kind="bar", rot=0
    )
    plt.xlabel("Experience group (low <4y, mid 4–9y, high 10+y)")
    plt.ylabel("Mean AI_Index")
    plt.title("Mean AI_Index by experience group")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "part2_ai_index_by_exp.png"))
    plt.close()

    # 2) By country group: only USA / UK / Other
    target_countries = [
        "United States of America",
        "United Kingdom of Great Britain and Northern Ireland",
        "Other",
    ]
    df_ai_country = df_ai[df_ai["CountryGroup"].isin(target_countries)].copy()
    usage_by_country = df_ai_country.groupby("CountryGroup")[
        ["AI_Index", "AI_TotalTasks"]
    ].mean()

    print("\n=== Part 2: mean AI usage by country (USA / UK / Other) ===")
    print(usage_by_country)
    usage_by_country.to_csv(
        os.path.join(OUTPUT_DIR, "part2_ai_usage_by_country_usa_uk_other.csv")
    )

    # Rename index for plotting
    label_map = {
        "United States of America": "USA",
        "United Kingdom of Great Britain and Northern Ireland": "UK",
        "Other": "Other",
    }
    usage_plot = usage_by_country.reindex(target_countries)
    usage_plot.index = [label_map[idx] for idx in usage_plot.index]

    plt.figure()
    usage_plot["AI_Index"].plot(kind="bar", rot=0)
    plt.xlabel("Country group")
    plt.ylabel("Mean AI_Index")
    plt.title("Mean AI_Index for USA / UK / Other countries")
    plt.tight_layout()
    plt.savefig(
        os.path.join(OUTPUT_DIR, "part2_ai_index_by_country_usa_uk_other.png")
    )
    plt.close()

    # 3) By age group
    age_order = [
        "18-24 years old",
        "25-34 years old",
        "35-44 years old",
        "45-54 years old",
        "55-64 years old",
        "65 years or older",
    ]
    usage_by_age = (
        df_ai.groupby("Age")[["AI_Index", "AI_TotalTasks"]]
        .mean()
        .reindex(age_order)
    )
    print("\n=== Part 2: mean AI usage by age group ===")
    print(usage_by_age)
    usage_by_age.to_csv(
        os.path.join(OUTPUT_DIR, "part2_ai_usage_by_age.csv")
    )

    plt.figure()
    usage_by_age["AI_Index"].plot(kind="bar", rot=45)
    plt.xlabel("Age group")
    plt.ylabel("Mean AI_Index")
    plt.title("Mean AI_Index by age group")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "part2_ai_index_by_age.png"))
    plt.close()


# Part 3: Age and salary (overall + by country)

def part3_age_and_salary(df: pd.DataFrame) -> None:
    """
    Part 3:
    - Median salary by age group (overall)
    - Median salary by age group within country (USA / UK / Other)
    """
    ensure_output_dir()

    age_order = [
        "18-24 years old",
        "25-34 years old",
        "35-44 years old",
        "45-54 years old",
        "55-64 years old",
        "65 years or older",
    ]

    # 1) Age vs salary (overall)
    salary_by_age = (
        df.groupby("Age")["ConvertedCompYearly"]
        .median()
        .reindex(age_order)
    )
    print("\n=== Part 3 (overall): median salary by age group ===")
    print(salary_by_age)
    salary_by_age.to_csv(
        os.path.join(OUTPUT_DIR, "part3_salary_by_age.csv")
    )

    plt.figure()
    salary_by_age.plot(kind="bar", rot=45)
    plt.xlabel("Age group")
    plt.ylabel("Median yearly compensation")
    plt.title("Median salary by age group (overall)")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "part3_salary_by_age.png"))
    plt.close()

    # 2) Age vs salary by country (USA / UK / Other)
    target_countries = [
        "United States of America",
        "United Kingdom of Great Britain and Northern Ireland",
        "Other",
    ]
    df_age_country = df[df["CountryGroup"].isin(target_countries)].copy()

    salary_age_country = (
        df_age_country.groupby(["Age", "CountryGroup"])["ConvertedCompYearly"]
        .median()
        .reset_index()
    )
    pivot = salary_age_country.pivot(
        index="Age", columns="CountryGroup", values="ConvertedCompYearly"
    ).reindex(age_order)
    pivot.to_csv(
        os.path.join(OUTPUT_DIR, "part3_salary_by_age_country_usa_uk_other.csv")
    )
    print("\n=== Part 3 (by country): median salary by age group (USA/UK/Other) ===")
    print(pivot)

    label_map = {
        "United States of America": "USA",
        "United Kingdom of Great Britain and Northern Ireland": "UK",
        "Other": "Other",
    }

    for country in target_countries:
        sub = salary_age_country[
            salary_age_country["CountryGroup"] == country
        ]
        sub = sub.set_index("Age").reindex(age_order).dropna()
        if sub.empty:
            continue
        plt.figure()
        sub["ConvertedCompYearly"].plot(kind="bar", rot=45)
        plt.xlabel("Age group")
        plt.ylabel("Median yearly compensation")
        plt.title(f"{label_map.get(country, country)}: median salary by age group")
        plt.tight_layout()
        safe_name = country.replace(" ", "_").replace("/", "_")
        fname = f"part3_salary_by_age_country_{safe_name}.png"
        plt.savefig(os.path.join(OUTPUT_DIR, fname))
        plt.close()

# Part 4: Role, salary and AI usage
def shorten_role(name: str) -> str:
    """
    Shorten long DevType role names to compact, readable labels for plots.
    """
    mapping: Dict[str, str] = {
        "Developer, back-end": "Back-end dev",
        "Developer, front-end": "Front-end dev",
        "Developer, full-stack": "Full-stack dev",
        "Developer, mobile": "Mobile dev",
        "Developer, desktop or enterprise applications": "Desktop/enterprise dev",
        "Developer, embedded applications or devices": "Embedded dev",
        "Developer, game or graphics": "Game/graphics dev",
        "Developer, QA or test": "QA/test dev",
        "Developer, data scientist or machine learning specialist": "Dev + DS/ML",
        "Data scientist or machine learning specialist": "DS/ML",
        "Database administrator": "DB admin",
        "DevOps specialist": "DevOps",
        "Engineer, data": "Data engineer",
        "Engineer, site reliability": "SRE",
        "Engineering manager": "Eng manager",
        "Product manager": "Product manager",
        "System administrator": "Sysadmin",
        "Academic researcher": "Researcher",
        "Scientist": "Scientist",
    }
    if name in mapping:
        return mapping[name]
    # Fallback: use the part before the first comma, or the original name
    return name.split(",")[0].strip() if "," in name else name


def part4_role_age_salary(df: pd.DataFrame) -> None:
    """
    Part 4:
    - Take the first DevType entry as primary role.
    - Among all respondents with salary and AI_Index, pick the 8 most frequent
      primary roles, skipping any role whose name starts with "Other".
    - For these roles, compute:
        * median yearly salary
        * mean AI_Index (overall AI usage intensity)
    - Save a CSV table and two bar plots:
        * Median salary by role
        * Mean AI_Index by role
    """
    ensure_output_dir()

    #  Build primary role and filter usable rows
    df_role = df.copy()

    def extract_primary_role(x: object) -> str:
        """Take the first DevType entry before ';' as primary role."""
        if pd.isna(x):
            return np.nan
        first = str(x).split(";")[0].strip()
        return first if first else np.nan

    df_role["PrimaryRole"] = df_role["DevType"].apply(extract_primary_role)

    # Keep rows with primary role, salary and AI_Index
    df_role = df_role[
        df_role["PrimaryRole"].notna()
        & df_role["ConvertedCompYearly"].notna()
        & df_role["AI_Index"].notna()
    ].copy()

    #  Pick top 8 roles by frequency, skipping "Other..." roles
    value_counts = df_role["PrimaryRole"].value_counts()

    top_roles: list[str] = []
    for role in value_counts.index:
        # Skip any role that starts with "Other" (case-insensitive)
        if role.strip().lower().startswith("other"):
            continue
        top_roles.append(role)
        if len(top_roles) >= 8:
            break

    df_top = df_role[df_role["PrimaryRole"].isin(top_roles)].copy()


    # Compute median salary and mean AI_Index for each role
    median_salary = (
        df_top.groupby("PrimaryRole")["ConvertedCompYearly"].median()
    )
    mean_ai_index = df_top.groupby("PrimaryRole")["AI_Index"].mean()

    stats_by_role = pd.DataFrame(
        {
            "MedianSalary": median_salary,
            "MeanAIIndex": mean_ai_index,
        }
    ).sort_values("MedianSalary")

    print(
        "\n=== Part 4: median salary and mean AI_Index "
        "(top 8 primary roles') ==="
    )
    print(stats_by_role)

    stats_by_role.to_csv(
        os.path.join(OUTPUT_DIR, "part4_role_salary_aiindex_stats.csv")
    )

    # Prepare shortened labels for plotting
    short_labels = [shorten_role(r) for r in stats_by_role.index]

    # Plot median salary by role
    plt.figure()
    ax1 = stats_by_role["MedianSalary"].plot(kind="bar", rot=30)
    ax1.set_xticklabels(short_labels, rotation=30, ha="right")
    plt.xlabel("Primary role (shortened)")
    plt.ylabel("Median yearly compensation")
    plt.title(
        "Median salary by primary role\n"
        "(top 8 primary roles)"
    )
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "part4_salary_by_role.png"))
    plt.close()

    # Plot mean AI_Index by role
    plt.figure()
    ax2 = stats_by_role["MeanAIIndex"].plot(kind="bar", rot=30)
    ax2.set_xticklabels(short_labels, rotation=30, ha="right")
    plt.xlabel("Primary role (shortened)")
    plt.ylabel("Mean AI_Index")
    plt.title(
        "Mean AI_Index by primary role\n"
        "(top 8 primary roles)"
    )
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "part4_ai_index_by_role.png"))
    plt.close()

# Part 5: Stepwise effects of AI use (no use vs partial vs mostly)
def part5_ai_use_intensity_steps(df: pd.DataFrame) -> None:
    """
    Part 5:
    Study whether partial AI use lies between non-use and mostly-use in terms
    of compensation, and whether there is a stepwise relationship between
    AI use intensity and salary.

    Steps:
    - Classify respondents into three AI use groups:
        * no_use:    no task is currently mostly/partially AI
        * partial_only: at least one task is currently partially AI, but none is mostly AI
        * mostly:   at least one task is currently mostly AI
    - Compare median yearly salary across the three groups.
    - Run a regression:
        log_salary ~ C(AI_UseGroup, reference='no_use') + controls
      to see how partial-only and mostly users differ from non-users.
    """
    ensure_output_dir()

    # Build AI_UseGroup
    df_step = df.copy()

    # Keep rows with salary, AI usage info, and log_salary
    df_step = df_step[
        df_step["log_salary"].notna()
        & df_step["AI_TotalTasks"].notna()
        & df_step["AI_NumMostly"].notna()
        & df_step["AI_NumPartial"].notna()
    ].copy()

    def classify_use(row):
        if row["AI_NumMostly"] > 0:
            return "mostly"
        elif row["AI_NumPartial"] > 0:
            return "partial_only"
        else:
            return "no_use"

    df_step["AI_UseGroup"] = df_step.apply(classify_use, axis=1)

    group_order = ["no_use", "partial_only", "mostly"]
    label_map = {
        "no_use": "No AI use",
        "partial_only": "Partial AI use only",
        "mostly": "Mostly AI use",
    }

    df_step["AI_UseGroup"] = pd.Categorical(
        df_step["AI_UseGroup"], categories=group_order, ordered=True
    )

    # Descriptive: median salary by AI use group
    salary_by_group = (
        df_step.groupby("AI_UseGroup")["ConvertedCompYearly"]
        .median()
        .reindex(group_order)
    )

    print("\n=== Part 5: median salary by AI use group ===")
    print(salary_by_group)

    salary_by_group.to_csv(
        os.path.join(OUTPUT_DIR, "part5_salary_by_ai_use_group.csv")
    )

    plt.figure()
    x_labels = [label_map[g] for g in group_order]
    plt.bar(x_labels, salary_by_group.values)
    plt.xlabel("AI use group")
    plt.ylabel("Median yearly compensation")
    plt.title("Median salary by AI use group\n(no use vs partial vs mostly)")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "part5_salary_by_ai_use_group.png"))
    plt.close()

    # Regression: log_salary ~ AI_UseGroup + controls
    df_reg = df_step[
        df_step["ExpGroup"].notna()
        & df_step["CountryGroup"].notna()
        & df_step["OrgSize"].notna()
        & df_step["RemoteWork"].notna()
        & df_step["EdLevel"].notna()
        & df_step["Age"].notna()
    ].copy()

    df_reg["AI_UseGroup"] = pd.Categorical(
        df_reg["AI_UseGroup"], categories=group_order, ordered=True
    )

    formula = (
        "log_salary ~ "
        "C(AI_UseGroup, Treatment(reference='no_use'))"
        " + C(ExpGroup)"
        " + C(CountryGroup)"
        " + C(OrgSize)"
        " + C(RemoteWork)"
        " + C(EdLevel)"
        " + C(Age)"
    )

    model = smf.ols(formula=formula, data=df_reg).fit()
    print(
        "\n=== Part 5: regression with AI use groups "
        "(baseline = no_use) ==="
    )
    print(model.summary())

    coef_table = model.summary2().tables[1]
    coef_table.to_csv(
        os.path.join(OUTPUT_DIR, "part5_regression_ai_use_group_coefs.csv")
    )

    # Stepwise interpretation in percentage terms
    for group, desc in [
        ("partial_only", "Partial users vs non-users"),
        ("mostly", "Mostly users vs non-users"),
    ]:
        beta = None
        for name in model.params.index:
            if f"[T.{group}]" in name:
                beta = model.params[name]
                break
        if beta is not None:
            pct = (np.exp(beta) - 1.0) * 100
            print(
                f"Interpretation: {desc}: coefficient ≈ {beta:.4f}, "
                f"which corresponds to about {pct:.2f}% difference in "
                f"predicted salary compared with non-users."
            )

# Main

def main() -> None:
    df = prepare_model_dataset()
    print("Modelling dataset shape:", df.shape)

    part1_ai_vs_salary(df)
    part2_ai_usage_profile(df)
    part3_age_and_salary(df)
    part4_role_age_salary(df)
    part5_ai_use_intensity_steps(df)


if __name__ == "__main__":
    main()
