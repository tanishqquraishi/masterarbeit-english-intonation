"""
___date__: 09 / 2024
__author__: Tanishq Quraishi

"""

import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
from matplotlib.ticker import FuncFormatter 
from utils import load_data, rename_columns, create_binary_column

"""
Investigate the likelihood of a high boundary tone by a speaker group
with factors such as formality and gender.

"""


# Load the data
file_path = r"C:\\Users\\Tanishq\\Documents\\stuttgart\\Study\\thesis\\data\\model data\\boundary_tones_17.07.2024.xlsx"
data = load_data(file_path)
data = rename_columns(data)

# Define lists for high and low boundary tones
high_boundary_tones = [
    'H-H%', 'L-H%', 'H-', 'H-^H%', 'L-^H%', '!H-H%', '^H-H%'
]
low_boundary_tones = [
    'L-L%', 'H-L%', 'L-', '!H-L%', '^H-L%'
]

# Create a binary column for high and low boundary tones 
def boundary_tone_condition(row):
    if row['boundary_tone'] in high_boundary_tones:
        return 1
    elif row['boundary_tone'] in low_boundary_tones:
        return 0
    return None

data = create_binary_column(data, 'boundary_tone_binary', boundary_tone_condition)

# Drop rows with None values in the boundary_tone_binary column
data = data.dropna(subset=['boundary_tone_binary'])

# Contrast-code the independent variables
data['bilingual_contrast'] = data['bilingual'].apply(lambda x: 1 if x == 'yes' else -1)
data['formality_contrast'] = data['formality'].apply(lambda x: 1 if x == 'formal' else -1)
data['gender_contrast'] = data['gender'].apply(lambda x: 1 if x == 'female' else -1)

# Interaction terms
data['interaction_formality_gender'] = data['formality_contrast'] * data['gender_contrast']
data['interaction_bilingual_gender'] = data['bilingual_contrast'] * data['gender_contrast']
data['interaction_bilingual_formality'] = data['bilingual_contrast'] * data['formality_contrast']

# Define the mixed effects model formula
formula = 'boundary_tone_binary ~ bilingual_contrast * formality_contrast + gender_contrast'

# Fit the Mixed Linear Model with random intercept for speaker_id
mixedlm_model = smf.mixedlm(formula, data, groups=data["speaker_id"]).fit()

# Print the summary of the MixedLM model
print(mixedlm_model.summary())

# Store the fitted values in the DataFrame
data['fittedvalues'] = mixedlm_model.fittedvalues

# Visualization of fixed effects coefficients with error bars
model_data = {
    'Variable': mixedlm_model.params.index,
    'Coef.': mixedlm_model.params.values,
    'Std.Err.': mixedlm_model.bse.values,
}

model_df = pd.DataFrame(model_data)

################## Visualizations ##################

# Visualize the coefficients
fig, ax = plt.subplots(figsize=(10, 6))

# Plotting the coefficients with error bars
ax.errorbar(model_df['Variable'], model_df['Coef.'], yerr=model_df['Std.Err.'], fmt='o', color='blue', ecolor='black', capsize=5)

# Add horizontal line at y=0 for reference
ax.axhline(y=0, color='grey', linestyle='--')

# Add labels and title
ax.set_xlabel('Variable')
ax.set_ylabel('Coefficient')
ax.set_title('MixedLM Coefficients with 95% Conf Interval for BT Binary')

# Rotate x-axis labels for better readability
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Plot the likelihood of high boundary tone by formality and gender
def plot_likelihood_by_group(data, group_col, title, ylabel):
    group_likelihood = data.groupby(group_col)['fittedvalues'].mean()

    # Create bar plot
    plt.figure(figsize=(8, 6))
    group_likelihood.plot(kind='bar', color=['skyblue', 'salmon'])
    plt.xlabel(group_col.capitalize())
    plt.ylabel(ylabel)
    plt.title(title)
    plt.xticks(rotation=0)  # Keep x-axis labels horizontal

    # Apply the percentage formatter to the y-axis
    plt.gca().yaxis.set_major_formatter(FuncFormatter(lambda y, _: f'{y:.0%}'))
    plt.tight_layout()
    plt.show()

# Plotting by formality
plot_likelihood_by_group(data, 'formality', 'Likelihood of High Boundary Tone by Formality', 'Likelihood of High Boundary Tone (%)')

# Plotting by gender
plot_likelihood_by_group(data, 'gender', 'Likelihood of High Boundary Tone by Gender', 'Likelihood of High Boundary Tone (%)')

# Post-hoc pairwise comparison
from statsmodels.stats.multicomp import pairwise_tukeyhsd

# Step 1: Create a new column combining gender and bilingualism with clearer names
def combine_gender_bilingual(row):
    if row['gender'] == 'male' and row['bilingual'] == 'yes':
        return 'male_bilingual'
    elif row['gender'] == 'male' and row['bilingual'] == 'no':
        return 'male_monolingual'
    elif row['gender'] == 'female' and row['bilingual'] == 'yes':
        return 'female_bilingual'
    elif row['gender'] == 'female' and row['bilingual'] == 'no':
        return 'female_monolingual'

data['gender_bilingual'] = data.apply(combine_gender_bilingual, axis=1)

# Step 2: Perform Tukey's HSD test for the combined effect of gender and bilingualism
tukey_combined = pairwise_tukeyhsd(
    endog=data['fittedvalues'],  # Dependent variable (fitted values)
    groups=data['gender_bilingual'],  # Independent variable (combined gender and bilingualism groups)
    alpha=0.05  # Significance level
)

# Step 3: Print the summary of the Tukey HSD test
print("Tukey HSD Test for Gender and Bilingualism Combined:")
print(tukey_combined.summary())

# Step 4: Plot the Tukey HSD results for the combined groups
tukey_combined.plot_simultaneous()

# Customize the plot for better readability
plt.xlabel('Mean Difference in Fitted Values')
plt.ylabel('Group (Gender and Bilingualism)')
plt.title('Tukey HSD Test: Pairwise Differences in Likelihood of High Boundary Tone')
plt.show()
