import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter
from statsmodels.stats.multicomp import pairwise_tukeyhsd
import pandas as pd
from tabulate import tabulate 

def plot_coefficients(model_df, title="GLMM Coefficients"):
    """
    Visualize the fixed effects coefficients with error bars.
    Parameters:
    model_df (pd.DataFrame): DataFrame containing model coefficients and standard errors.
        - Columns should include 'Variable' (names of variables), 'Coef.' (coefficients), and 'Std.Err.' (standard errors).
    title (str): Title for the plot.

    Returns:
    None: Displays a matplotlib plot.
    """
    #fig check 
    ax = plt.subplots(figsize=(10, 6))

    # Plot coefficients with error bars
    ax.errorbar(model_df['Variable'], model_df['Coef.'], yerr=model_df['Std.Err.'], fmt='o', color='blue', ecolor='black', capsize=5)

    # Add horizontal line at y=0 for reference
    ax.axhline(y=0, color='grey', linestyle='--')

    # Add labels and title
    ax.set_xlabel('Variable')
    ax.set_ylabel('Coefficient')
    ax.set_title(title)

    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

import matplotlib.pyplot as plt

def plot_coefficients_1(model_df, title="GLMM Coefficients"):
    """
    Visualize the fixed effects coefficients with confidence intervals.
    
    Parameters:
    model_df (pd.DataFrame): DataFrame containing model coefficients.
        - Must include 'Variable' (names of variables), 'Coef.' (coefficients),
          '2.5_ci' (lower bound), '97.5_ci' (upper bound).
    title (str): Title for the plot.

    Returns:
    None: Displays a matplotlib plot.
    """
    # Ensure correct unpacking of plt.subplots()
    fig, ax = plt.subplots(figsize=(10, 6))  

    # Compute confidence interval range
    lower_error = model_df['Coef.'] - model_df['2.5_ci']
    upper_error = model_df['97.5_ci'] - model_df['Coef.']
    errors = [lower_error, upper_error]

    # Sort by coefficient magnitude for better visualization
    model_df = model_df.sort_values(by='Coef.', ascending=False)

    # Plot coefficients with error bars
    ax.errorbar(model_df['Variable'], model_df['Coef.'], yerr=errors, fmt='o', color='blue', ecolor='black', capsize=5)

    # Add horizontal line at y=0 for reference
    ax.axhline(y=0, color='grey', linestyle='--')

    # Labels and title
    ax.set_xlabel('Variable')
    ax.set_ylabel('Coefficient')
    ax.set_title(title)

    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()



def plot_likelihood_by_group(data, group_col, title, ylabel):
    """
    Plot the likelihood of X (pitch accents or boundary tones) by group.

    Parameters:
    data (pd.DataFrame): DataFrame containing fitted values and grouping variables.
    group_col (str): Column name to group data by ('gender' or 'formality').
    title (str): Title for the plot.
    ylabel (str): Label for the y-axis.

    Returns:
    None: Displays a matplotlib bar plot.
    """
    group_likelihood = data.groupby(group_col)['fittedvalues'].mean()

    plt.figure(figsize=(8, 6))
    group_likelihood.plot(kind='bar', color=['skyblue', 'pink'])
    plt.xlabel(group_col.capitalize())
    plt.ylabel(ylabel)
    plt.title(title)
    plt.xticks(rotation=0)
    plt.gca().yaxis.set_major_formatter(FuncFormatter(lambda y, _: f'{y:.0%}'))
    plt.tight_layout()
    plt.show()

def plot_likelihood_by_interaction(data, interaction_vars, title, ylabel):
    """
    Plot the interaction effect between two variables on the likelihood of pitch accents or boundary tones.
    interaction_vars: List containing two columns to plot interaction (e.g., ['bilingual', 'gender'])

    Parameters:
    data (pd.DataFrame): DataFrame containing interaction variables.
    interaction_vars (list): List containing two columns to plot interaction (e.g., ['bilingual', 'gender']).
    title (str): Title for the plot.
    ylabel (str): Label for the y-axis.

    Returns:
    None: Displays a matplotlib bar plot showing interaction effects.
    """
    # Group data by interaction variables and calculate mean fitted values
    interaction_data = data.groupby(interaction_vars)['fittedvalues'].mean().unstack()

    # Create a bar plot for the interaction
    fig, ax = plt.subplots(figsize=(8, 6))
    interaction_data.plot(kind='bar', ax=ax)

    # Add labels and title
    ax.set_xlabel(interaction_vars[0].capitalize())
    ax.set_ylabel(ylabel)
    ax.set_title(title)

    # Format y-axis as percentage
    ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f'{y:.0%}'))
    
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.show()

def plot_likelihood_by_group_1(data, group_col, title, ylabel):
    """
    Plot the likelihood of a high boundary tone by group with confidence intervals.

    Parameters:
    data (pd.DataFrame): DataFrame containing fitted values and grouping variables.
    group_col (str): Column name to group data by ('gender' or 'formality').
    title (str): Title for the plot.
    ylabel (str): Label for the y-axis.

    Returns:
    None: Displays a matplotlib bar plot with error bars.
    """
    # Compute mean, standard deviation, and confidence intervals
    group_stats = data.groupby(group_col)['fittedvalues'].agg(['mean', 'std', 'count'])

    # Compute 95% confidence intervals
    group_stats['ci'] = 1.96 * (group_stats['std'] / np.sqrt(group_stats['count']))

    # Create bar plot with error bars
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.bar(group_stats.index, group_stats['mean'], yerr=group_stats['ci'], capsize=5, color=['skyblue', 'salmon'])

    # Annotate bars with actual likelihood values
    for i, value in enumerate(group_stats['mean']):
        ax.text(i, value + 0.02, f"{value:.2f}", ha='center', fontsize=12)

    # Labels and title
    ax.set_xlabel(group_col.capitalize())
    ax.set_ylabel(ylabel)
    ax.set_title(title)

    # Format y-axis as percentage
    ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f'{y:.0%}'))

    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.show()


def plot_mean_with_ci(data, group_col, title, ylabel):
    """
    Plot mean values with confidence intervals for categorical variables.

    Parameters:
    data (pd.DataFrame): DataFrame containing fitted values and grouping variables.
    group_col (str): Column name to group data by (e.g., 'gender' or 'formality').
    title (str): Title for the plot.
    ylabel (str): Label for the y-axis.

    Returns:
    None: Displays a matplotlib error bar plot with means and confidence intervals.
    """
    # Compute mean, standard deviation, and confidence intervals
    group_stats = data.groupby(group_col)['fittedvalues'].agg(['mean', 'std', 'count'])

    # Compute 95% confidence intervals
    group_stats['ci'] = 1.96 * (group_stats['std'] / np.sqrt(group_stats['count']))

    # Create error bar plot
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.errorbar(group_stats.index, group_stats['mean'], yerr=group_stats['ci'], fmt='o', capsize=5, color='black', markersize=8)

    # Print values for debugging
    print(group_stats)

    # Annotate points with actual mean values
    #for i, value in enumerate(group_stats['mean']):
    #    ax.text(i, value + 0.01, f"{value:.2f}", ha='center', fontsize=12)

    # Labels and title
    ax.set_xlabel(group_col.capitalize())
    ax.set_ylabel(ylabel)
    ax.set_title(title)

    # Format y-axis as percentage
    ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f'{y:.0%}'))

    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.show()



def plot_tukey_hsd(data, group_col, value_col, title="Multiple Comparisons Between All Pairs (Tukey)"):
    """
    Perform Tukey HSD test and plot multiple comparisons with confidence intervals.
    
    Parameters:
    data (pd.DataFrame): DataFrame containing the data.
    group_col (str): Column name that contains categorical groups (e.g., 'bilingual_formality').
    value_col (str): Column name with numerical values (e.g., 'ip_length').
    title (str): Title for the plot.
    
    Returns:
    None: Displays a matplotlib dot plot with confidence intervals.
    """
    # Perform Tukey HSD test
    tukey_results = pairwise_tukeyhsd(endog=data[value_col], groups=data[group_col], alpha=0.05)
    
    # Extract group names, means, and confidence intervals
    group_means = data.groupby(group_col)[value_col].mean()
    ci_low = tukey_results.confint[:, 0]
    ci_high = tukey_results.confint[:, 1]
    groups = tukey_results.groupsunique
    
    # Sort groups for better visualization
    sorted_groups = sorted(groups, key=lambda g: group_means[g])
    
    # Extract means in sorted order
    sorted_means = [group_means[g] for g in sorted_groups]
    sorted_ci_low = [ci_low[list(groups).index(g)] for g in sorted_groups]
    sorted_ci_high = [ci_high[list(groups).index(g)] for g in sorted_groups]
    
    # Compute error bars
    errors = [np.array(sorted_means) - np.array(sorted_ci_low), np.array(sorted_ci_high) - np.array(sorted_means)]
    
    # Plot the results
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.errorbar(sorted_means, sorted_groups, xerr=errors, fmt='o', color='black', capsize=5)
    ax.set_xlabel("Average Number of Words")
    ax.set_title(title)
    
    plt.tight_layout()
    plt.show()


import pandas as pd
from tabulate import tabulate

def display_model_fit(glmm_model):
    """
    Display model fit statistics as a formatted table.
    
    Parameters:
    glmm_model (Lmer): The fitted GLMM model.
    
    Returns:
    None: Prints a formatted table.
    """
    # Infer model family from pymer4 output
    model_family = glmm_model.family if hasattr(glmm_model, "family") else "Unknown"

    model_fit_info = {
        "Model Family": [model_family],  
        "Inference": ["Parametric"],
        "Observations": [glmm_model.data.shape[0]],
        "Groups": [glmm_model.grps],  # Corrected attribute for groups
        "Log-Likelihood": [round(glmm_model.logLike, 3)],
        "AIC": [round(glmm_model.AIC, 3)]
    }
    
    fit_df = pd.DataFrame(model_fit_info)
    
    print("\nModel Fit Summary:")
    print(tabulate(fit_df, headers="keys", tablefmt="pretty"))


import pandas as pd
from tabulate import tabulate

def display_fixed_effects(glmm_model):
    """
    Display the fixed effects coefficients as a formatted table.
    
    Parameters:
    glmm_model (Lmer): The fitted GLMM model.
    
    Returns:
    None: Prints a formatted table.
    """
    # Extract base fixed effects coefficients
    fixed_effects_table = glmm_model.coefs[['Estimate', '2.5_ci', '97.5_ci', 'SE']].copy()

    # Check whether Z-stat or T-stat exists
    if 'Z-stat' in glmm_model.coefs.columns:
        fixed_effects_table['Z-stat'] = glmm_model.coefs['Z-stat']
    elif 'T-stat' in glmm_model.coefs.columns:  # Gaussian models use T-stat instead
        fixed_effects_table['T-stat'] = glmm_model.coefs['T-stat']

    # Check if P-val exists before applying formatting
    if 'P-val' in glmm_model.coefs.columns:
        fixed_effects_table['P-val'] = glmm_model.coefs['P-val'].apply(lambda x: "<0.001" if x < 0.001 else round(x, 3)).astype(str)

    # Reset index for readability
    fixed_effects_table.reset_index(inplace=True)
    fixed_effects_table.rename(columns={"index": "Effect"}, inplace=True)

    print("\nFixed Effects Coefficients:")
    print(tabulate(fixed_effects_table, headers="keys", tablefmt="pretty"))
