import pandas as pd
from config import file_paths
from pymer4.models import Lmer
from utils import load_data, rename_columns, create_binary_column
from visualizations import plot_mean_with_ci, plot_coefficients_1

"""
Investigate the likelihood of a high BT by each speaker group (bilingual vs. monolingual speakers) 
with factors such as formality and gender.
"""

# Load data
file_path = file_paths["bt_model"]
data = load_data(file_path)
data = rename_columns(data)

# Define lists for high and low BT
high_boundary_tones = ['H-H%', 'L-H%', 'H-', 'H-^H%', 'L-^H%', '!H-H%', '^H-H%']
low_boundary_tones = ['L-L%', 'H-L%', 'L-', '!H-L%', '^H-L%']

# Create a binary column for high and low BT
data = create_binary_column(data, 'boundary_tone_binary', lambda row: 1 if row['boundary_tone'] in high_boundary_tones else 0)

# Drop rows with None values
data = data.dropna(subset=['boundary_tone_binary'])

# Contrast-code the independent variables
data['bilingual_contrast'] = data['bilingual'].apply(lambda x: 1 if x == 'yes' else -1)
data['formality_contrast'] = data['formality'].apply(lambda x: 1 if x == 'formal' else -1)
data['gender_contrast'] = data['gender'].apply(lambda x: 1 if x == 'female' else -1)

# Interaction terms
data['interaction_formality_gender'] = data['formality_contrast'] * data['gender_contrast']
data['interaction_bilingual_gender'] = data['bilingual_contrast'] * data['gender_contrast']
data['interaction_bilingual_formality'] = data['bilingual_contrast'] * data['formality_contrast']

# Define the generalized linear mixed model using pymer4
formula = 'boundary_tone_binary ~ bilingual_contrast + formality_contrast + gender_contrast + interaction_bilingual_formality + interaction_bilingual_gender + interaction_formality_gender + (1|speaker_id)'

# Fit the GLMM model using a binomial family
glmm_model = Lmer(formula, data=data, family='binomial')
glmm_model.fit()

# Print the fixed effects coeffs
print("Fixed Effects Coefficients:")
print(glmm_model.coefs)


# Store the fitted values in the DataFrame
data['fittedvalues'] = glmm_model.predict(data, skip_data_checks=True, verify_predictions=False)

######## Extract z and p scores for reporting #############

# Extract Z-stat and P-val from the model coefficients
z_and_p_values = glmm_model.coefs[['Z-stat', 'P-val']]

# Rename the index for better readability
z_and_p_values.index.name = 'Effect'

# Customize p-value formatting
z_and_p_values['P-val'] = z_and_p_values['P-val'].apply(lambda x: "<0.001" if x < 0.001 else round(x, 3))
print("Z-scores and P-values for Fixed Effects: ")
print(z_and_p_values)

################## Visualizations ##################
# Prepare the model data for visualizations
model_data = {
    'Variable': glmm_model.coefs['Estimate'].index,
    'Coef.': glmm_model.coefs['Estimate'].values,
    '2.5_ci': glmm_model.coefs['2.5_ci'].values,  # Lower confidence bound
    '97.5_ci': glmm_model.coefs['97.5_ci'].values,  # Upper confidence bound
    'Std.Err.': glmm_model.coefs['SE'].values,
}

model_df = pd.DataFrame(model_data)

plot_mean_with_ci(data, 'formality', 'Likelihood of High Boundary Tone by Formality', 'Likelihood of High Boundary Tone')
plot_mean_with_ci(data, 'gender', 'Likelihood of High Boundary Tone by Gender', 'Likelihood of High Boundary Tone')

plot_coefficients_1(model_df, title='GLMM Coefficients for Boundary Tones')


#old
#plot_coefficients(model_df, title='GLMM Coefficients for Boundary Tones')
#plot_likelihood_by_group(data, 'formality', 'Likelihood of High Boundary Tone by Formality', 'Likelihood of High Boundary Tone')
#plot_likelihood_by_group(data, 'gender', 'Likelihood of High Boundary Tone by Gender', 'Likelihood of High Boundary Tone')
#plot_likelihood_by_interaction(data, ['bilingual', 'gender'], 'Interaction of Bilingualism and Gender on High Boundary Tone', 'Likelihood')