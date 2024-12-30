import pandas as pd
from pymer4.models import Lmer
from utils import load_data, rename_columns, create_binary_column
from visualizations_binomial_models import plot_coefficients, plot_likelihood_by_group, plot_likelihood_by_interaction

# Load data
file_path = r"C:\\Users\\Tanishq\\Documents\\stuttgart\\Study\\thesis\\data\\model data\\boundary_tones_17.07.2024.xlsx"
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

# Prepare the model data for visualizations
model_data = {
    'Variable': glmm_model.coefs['Estimate'].index,
    'Coef.': glmm_model.coefs['Estimate'].values,
    'Std.Err.': glmm_model.coefs['SE'].values,
}

model_df = pd.DataFrame(model_data)

# Visualizations
plot_coefficients(model_df, title='GLMM Coefficients for Boundary Tones')
plot_likelihood_by_group(data, 'formality', 'Likelihood of High Boundary Tone by Formality', 'Likelihood of High Boundary Tone (%)')
plot_likelihood_by_group(data, 'gender', 'Likelihood of High Boundary Tone by Gender', 'Likelihood of High Boundary Tone (%)')
plot_likelihood_by_interaction(data, ['bilingual', 'gender'], 'Interaction of Bilingualism and Gender on High Boundary Tone', 'Likelihood (%)')