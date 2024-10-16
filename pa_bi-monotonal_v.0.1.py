import pandas as pd
from pymer4.models import Lmer
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from utils import load_data, rename_columns, create_binary_column

# Load the data
file_path = r"C:\\Users\\Tanishq\\Documents\\stuttgart\\Study\\thesis\\data\\model data\\pitch_accents_17.07.2024.xlsx"
data = load_data(file_path)
data = rename_columns(data)

# Define monotonal and bitonal pitch accents
monotonal_pitch_accents = ["H*", "L*"]
bitonal_pitch_accents = ["L+H*", "L*+H", "H+L*", "H*+!H", "H*+L", "H+!H*", "H*+L"]

# Create a binary column for pitch accents (1 = monotonal, 0 = bitonal)
data = create_binary_column(
    data, 
    'pa_type_binary', 
    lambda row: 1 if row['word_pa'] in monotonal_pitch_accents else 
                (0 if row['word_pa'] in bitonal_pitch_accents else None)
)

# Drop rows with None values in the pa_type_binary column
data = data.dropna(subset=['pa_type_binary'])

# Contrast-code the independent variables
data['bilingual_contrast'] = data['bilingual'].apply(lambda x: 1 if x == 'yes' else -1)
data['formality_contrast'] = data['formality'].apply(lambda x: 1 if x == 'formal' else -1)
data['gender_contrast'] = data['gender'].apply(lambda x: 1 if x == 'female' else -1)

# Manually create interaction terms
data['interaction_formality_gender'] = data['formality_contrast'] * data['gender_contrast']
data['interaction_bilingual_gender'] = data['bilingual_contrast'] * data['gender_contrast']
data['interaction_bilingual_formality'] = data['bilingual_contrast'] * data['formality_contrast']

# Define the GLMM formula using manual interaction terms
formula = 'pa_type_binary ~ bilingual_contrast + formality_contrast + gender_contrast + interaction_bilingual_formality + interaction_bilingual_gender + interaction_formality_gender + (1|speaker_id)'

# Fit the GLMM model using a binomial family
glmm_model = Lmer(formula, data=data, family='binomial')
glmm_model.fit()

# Print the summary of the GLMM model
print(glmm_model.summary())

# Store the fitted values in the DataFrame
data['fittedvalues'] = glmm_model.predict(data, skip_data_checks=True, verify_predictions=False)

################## Visualizations ##################

# Visualization of fixed effects coefficients with error bars
model_data = {
    'Variable': glmm_model.coefs['Estimate'].index,
    'Coef.': glmm_model.coefs['Estimate'].values,
    'Std.Err.': glmm_model.coefs['SE'].values,
}

model_df = pd.DataFrame(model_data)

# Visualize the coefficients
fig, ax = plt.subplots(figsize=(12, 8))

# Plotting the coefficients with error bars
ax.errorbar(model_df['Variable'], model_df['Coef.'], yerr=model_df['Std.Err.'], fmt='o', color='blue', ecolor='black', capsize=5)

# Add horizontal line at y=0 for reference
ax.axhline(y=0, color='grey', linestyle='--')

# Add labels and title
ax.set_xlabel('Variable')
ax.set_ylabel('Coefficient')
ax.set_title('GLMM Coefficients with 95% Conf Interval for Bitonal vs Monotonal PA')

# Rotate x-axis labels for better readability
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Plot the likelihood of a bitonal PA based on speaker group, formality, and gender
data['predicted_pa'] = data['fittedvalues']

def plot_likelihood_by_group(data, group_col, title, ylabel, hue=None):
    group_likelihood = data.groupby(group_col)['predicted_pa'].mean()

    # Create bar plot
    plt.figure(figsize=(10, 6))
    if hue:
        group_likelihood = data.groupby([group_col, hue])['predicted_pa'].mean().unstack()
        group_likelihood.plot(kind='bar', color=['skyblue', 'salmon'])
    else:
        group_likelihood.plot(kind='bar', color=['skyblue', 'salmon'])
        
    plt.xlabel(group_col.capitalize())
    plt.ylabel(ylabel)
    plt.title(title)
    plt.xticks(rotation=0)  # Keep x-axis labels horizontal

    # Apply the percentage formatter to the y-axis
    plt.gca().yaxis.set_major_formatter(FuncFormatter(lambda y, _: f'{y:.0%}'))
    plt.tight_layout()
    plt.show()

# Plotting by speaker group (bilingualism)
plot_likelihood_by_group(data, 'bilingual', 'Likelihood of Bitonal Pitch Accent by Speaker Group', 'Likelihood of Bitonal Pitch Accent (%)')

# Plotting by formality with speaker group as hue
plot_likelihood_by_group(data, 'formality', 'Likelihood of Bitonal Pitch Accent by Formality and Speaker Group', 'Likelihood of Bitonal Pitch Accent (%)', hue='bilingual')

# Plotting by gender with speaker group as hue
plot_likelihood_by_group(data, 'gender', 'Likelihood of Bitonal Pitch Accent by Gender and Speaker Group', 'Likelihood of Bitonal Pitch Accent (%)', hue='bilingual')