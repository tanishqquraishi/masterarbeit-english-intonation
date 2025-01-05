import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

def plot_coefficients(model_df, title="GLMM Coefficients"):
    """
    Visualize the fixed effects coefficients with error bars.
    """
    fig, ax = plt.subplots(figsize=(10, 6))

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
    Plot the likelihood of pitch accents or boundary tones by group.
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