import pandas as pd
from config import file_paths

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning) # comment out to see warning regarding pymer4 version deprecation

from pymer4.models import Lmer
from utils import load_data, rename_columns, create_binary_column, convert_mixed_columns_to_string
from visualizations import plot_coefficients_1, plot_mean_with_ci, display_model_fit, display_fixed_effects

"""
Investigate the likelihood of a PA on a content word by each speaker group (bilingual vs. monolingual speakers) 
with factors such as formality.
"""

# Load data
file_path = file_paths["pa_con_model"]
data = load_data(file_path)
data = rename_columns(data)
# Convert mixed-type columns to string
data = convert_mixed_columns_to_string(data)


# Create a binary column for the presence or absence of word_pa
data = create_binary_column(data, 'word_pa_binary', lambda row: 1 if pd.notnull(row['word_pa']) else 0)

# Contrast-code the independent variables
data['bilingual_contrast'] = data['bilingual'].apply(lambda x: 1 if x == 'yes' else -1)
data['formality_contrast'] = data['formality'].apply(lambda x: 1 if x == 'formal' else -1)

# Define the mixed-effects model formula using pymer4 (with random intercept for speaker_id)
formula = 'word_pa_binary ~ bilingual_contrast * formality_contrast + (1|speaker_id)'

# Fit the GLMM model using a binomial family
glmm_model = Lmer(formula, data=data, family='binomial')
glmm_model.fit()

# Print the fixed effects coeffs
# print("Fixed Effects Coefficients:")
# print(glmm_model.coefs)

# Print the model summary
print(glmm_model.summary())

# Display Model Fit Statistics
display_model_fit(glmm_model)

# Display Fixed Effects Coefficients
display_fixed_effects(glmm_model)

# Store the fitted values in the DataFrame
data['fittedvalues'] = glmm_model.predict(data, skip_data_checks=True, verify_predictions=False)

######## Extract z and p scores for reporting #############

# Extract Z-stat and P-val from the model coefficients
z_and_p_values = glmm_model.coefs[['Z-stat', 'P-val']]

# Rename the index for better readability
z_and_p_values.index.name = 'Effect'

# Customize p-value formatting
z_and_p_values.loc[:, 'P-val'] = z_and_p_values['P-val'].apply(lambda x: "<0.001" if x < 0.001 else round(x, 3))
print("Z-scores and P-values for Fixed Effects Extracted for Reporting:")
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

plot_coefficients_1(model_df, title='GLMM Coefficients for Pitch Accents on Content Words')
plot_mean_with_ci(data, 'bilingual', 'Likelihood of Pitch Accent on Content Words by Speaker Group', 'Likelihood of Pitch Accent on a Content Word')
plot_mean_with_ci(data, 'formality', 'Likelihood of Pitch Accent on Content Words by Formality', 'Likelihood of Pitch Accent on a Content Word')