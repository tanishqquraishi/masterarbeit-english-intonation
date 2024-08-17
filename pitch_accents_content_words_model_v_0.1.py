import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
from matplotlib.ticker import FuncFormatter 
from utils import load_data, rename_columns, create_binary_column

# Load the data
file_path = r"C:\Users\Tanishq\Documents\stuttgart\Study\thesis\data\model data\pa_con_17.07.2024.xlsx"
data = load_data(file_path)

# Step 1: Create a binary column for the presence or absence of word_pa
data = create_binary_column(data, 'word_pa_binary', lambda row: 1 if pd.notnull(row['2_anno_default_ns:word_pa']) else 0)

# Step 2: Rename the columns for clarity
data = rename_columns(data)

# Contrast-code the independent variables
data['bilingual_contrast'] = data['bilingual'].apply(lambda x: 1 if x == 'yes' else -1)
data['formality_contrast'] = data['formality'].apply(lambda x: 1 if x == 'formal' else -1)
data['gender_contrast'] = data['gender'].apply(lambda x: 1 if x == 'female' else -1)

# Step 3: Define the mixed effects model formula
formula = 'word_pa_binary ~ bilingual_contrast * formality_contrast + gender_contrast'

# Fit the Mixed Linear Model with random intercept for speaker_id
mixedlm_model = smf.mixedlm(formula, data, groups=data["speaker_id"]).fit()

# Print the summary of the MixedLM model
print(mixedlm_model.summary())

# Step 4: Plot the model results
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
ax.set_title('MixedLM Coefficients with 95% Conf Interval for word_pa Binary')

# Rotate x-axis labels for better readability
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Step 5: Plot the likelihood of a word_pa based on monolingual and bilingual speaker groups
# Calculate the mean likelihood of word_pa by speaker group
data['predicted_pa'] = mixedlm_model.fittedvalues
group_likelihood = data.groupby('bilingual')['predicted_pa'].mean()

# Create bar plot for speaker group likelihood
plt.figure(figsize=(8, 6))
group_likelihood.plot(kind='bar', color=['skyblue', 'salmon'])
plt.xlabel('Speaker Group')
plt.ylabel('Likelihood of word_pa (%)')
plt.title('Likelihood of Presence of word_pa by Speaker Group')
plt.xticks(rotation=0)

# Apply the percentage formatter to the y-axis
plt.gca().yaxis.set_major_formatter(FuncFormatter(lambda y, _: f'{y:.0%}'))
plt.tight_layout()
plt.show()