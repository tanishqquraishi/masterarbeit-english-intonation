import matplotlib.pyplot as plt
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