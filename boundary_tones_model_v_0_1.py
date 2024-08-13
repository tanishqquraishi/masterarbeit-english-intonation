import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
import statsmodels.formula.api as smf
import data 

file_path = r"C:\Users\Tanishq\Documents\stuttgart\Study\thesis\data\model data\boundary_tones_17.07.2024.xlsx"
data = pd.read_excel(file_path)

# Rename the columns to more straightforward names
data = data.rename(columns={
    '1_anno_default_ns:bt': 'boundary_tone',
    '1_meta_speaker-bilingual': 'bilingual',
    '1_meta_setting': 'formality',
    '1_meta_speaker-gender': 'gender',
    '1_meta_speaker-id': 'speaker_id'
})

# Define lists for high and low boundary tones
high_boundary_tones = [
    'H-H%', 'L-H%', 'H-', 'H-^H%', 'L-^H%', '!H-H%', '^H-H%'
]
low_boundary_tones = [
    'L-L%', 'H-L%', 'L-', '!H-L%', '^H-L%'
]

# Create a binary column for high and low boundary tones
data['boundary_tone_binary'] = data['boundary_tone'].apply(
    lambda x: 1 if x in high_boundary_tones else (0 if x in low_boundary_tones else None)
)

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

# Visualization of fixed effects coefficients with error bars
model_data = {
    'Variable': mixedlm_model.params.index,
    'Coef.': mixedlm_model.params.values,
    'Std.Err.': mixedlm_model.bse.values,
}

model_df = pd.DataFrame(model_data)

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
    group_likelihood = data.groupby(group_col)[mixedlm_model.fittedvalues.name].mean()

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
