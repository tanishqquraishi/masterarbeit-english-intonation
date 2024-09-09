"""
___date__: 08 / 2024
__author__: Tanishq Quraishi
"""

import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
from matplotlib.ticker import FuncFormatter
from utils import load_data, rename_columns, create_binary_column

"""
Investigate the likelihood of a pitch accent on a content word 
by a speaker group with factors such as formality.
"""

# Load the data
file_path = r"C:\Users\Tanishq\Documents\stuttgart\Study\thesis\data\model data\pa_con_17.07.2024.xlsx"
data = load_data(file_path)
data = rename_columns(data)

# Create a binary column for the presence or absence of word_pa
data = create_binary_column(data, 'word_pa_binary', lambda row: 1 if pd.notnull(row['word_pa']) else 0)

# Contrast-code the independent variables
data['bilingual_contrast'] = data['bilingual'].apply(lambda x: 1 if x == 'yes' else -1)
data['formality_contrast'] = data['formality'].apply(lambda x: 1 if x == 'formal' else -1)

# Define the mixed effects model formula with only bilingualism and formality
formula = 'word_pa_binary ~ bilingual_contrast * formality_contrast'

# Fit the Mixed Linear Model with random intercept for speaker_id
mixedlm_model = smf.mixedlm(formula, data, groups=data["speaker_id"]).fit()

# Print the summary of the MixedLM model
print(mixedlm_model.summary())

################## Visualizations ##################

# Visualization of fixed effects coefficients with error bars
model_data = {
    'Variable': mixedlm_model.params.index,
    'Coef.': mixedlm_model.params.values,
    'Std.Err.': mixedlm_model.bse.values,
}

model_df = pd.DataFrame(model_data)

# Visualize the coefficients with enhanced readability
fig, ax = plt.subplots(figsize=(10, 6))

# Use colors only for the markers, error bars in a single color
colors = ['green' if coef > 0 else 'red' for coef in model_df['Coef.']]

# Plotting the coefficients with error bars
ax.errorbar(model_df['Variable'], model_df['Coef.'], yerr=model_df['Std.Err.'], fmt='o', color='black', ecolor='black', capsize=5)

# Plot the colored markers separately
for i, (coef, color) in enumerate(zip(model_df['Coef.'], colors)):
    ax.plot(model_df['Variable'][i], coef, 'o', color=color)

# Add horizontal line at y=0 for reference
ax.axhline(y=0, color='grey', linestyle='--')

# Add labels and title
ax.set_xlabel('Variable')
ax.set_ylabel('Coefficient')
ax.set_title('MixedLM Coefficients with 95% Confidence Interval')

# Add annotations to each point
for i, txt in enumerate(model_df['Coef.']):
    ax.annotate(f'{txt:.2f}', (model_df['Variable'][i], model_df['Coef.'][i]), textcoords="offset points", xytext=(0, 10), ha='center')

plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Plot the likelihood of a word_pa based on monolingual and bilingual speaker groups
# Calculate the mean likelihood of word_pa by speaker group
data['predicted_pa'] = mixedlm_model.fittedvalues
group_likelihood = data.groupby('bilingual')['predicted_pa'].mean()

# Create bar plot for speaker group likelihood
plt.figure(figsize=(8, 6))
group_likelihood.plot(kind='bar', color=['skyblue', 'salmon'])
plt.xlabel('Bilingual')
plt.ylabel('Likelihood of Pitch Accent on Content Word (%)')
plt.title('Likelihood of Presence of Pitch Accent on Content Word by Speaker Group')
plt.xticks(rotation=0)

# Apply the percentage formatter to the y-axis
plt.gca().yaxis.set_major_formatter(FuncFormatter(lambda y, _: f'{y:.0%}'))
plt.tight_layout()
plt.show()

# Calculate the mean likelihood of word_pa by formality
formality_likelihood = data.groupby('formality')['predicted_pa'].mean()

# Create bar plot for formality likelihood
plt.figure(figsize=(8, 6))
formality_likelihood.plot(kind='bar', color=['skyblue', 'salmon'])
plt.xlabel('Formality')
plt.ylabel('Likelihood of Pitch Accent on Content Word (%)')
plt.title('Likelihood of Presence of Pitch Accent on Content Word by Formality')
plt.xticks(rotation=0)

# Apply the percentage formatter to the y-axis
plt.gca().yaxis.set_major_formatter(FuncFormatter(lambda y, _: f'{y:.0%}'))
plt.tight_layout()
plt.show()