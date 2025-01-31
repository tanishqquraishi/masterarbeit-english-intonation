import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter


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