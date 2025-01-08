import pandas as pd
from pymer4.models import Lmer
from utils import load_data, rename_columns, create_binary_column
from visualizations_binomial_models import plot_coefficients, plot_likelihood_by_group

# Load data
file_path = r"C:\\Users\\Tanishq\\Documents\\stuttgart\\Study\\thesis\\data\\model data\\pa_con_for_model.xlsx"
data = load_data(file_path)
data = rename_columns(data)

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
print("Fixed Effects Coefficients:")
print(glmm_model.coefs)

# Store the fitted values in the DataFrame
data['fittedvalues'] = glmm_model.predict(data, skip_data_checks=True, verify_predictions=False)

# Extract only z and p scores in a table

# Extract Z-stat and P-val from the model coefficients
z_and_p_values = glmm_model.coefs[['Z-stat', 'P-val']]

# Rename the index for better readability
z_and_p_values.index.name = 'Effect'

# Customize p-value formatting
z_and_p_values['P-val'] = z_and_p_values['P-val'].apply(lambda x: "<0.001" if x < 0.001 else round(x, 3))
print("Z-scores and P-values for Fixed Effects: ")
print(z_and_p_values)

# Prepare the model data for visualizations
model_data = {
    'Variable': glmm_model.coefs['Estimate'].index,
    'Coef.': glmm_model.coefs['Estimate'].values,
    'Std.Err.': glmm_model.coefs['SE'].values,
}

model_df = pd.DataFrame(model_data)

# Visualizations
plot_coefficients(model_df, title='GLMM Coefficients for Pitch Accents on Content Words')
plot_likelihood_by_group(data, 'bilingual', 'Likelihood of Pitch Accent on Content Words by Speaker Group', 'Likelihood of Pitch Accent on a Content Word')
plot_likelihood_by_group(data, 'formality', 'Likelihood of Pitch Accent on Content Words by Formality', 'Likelihood of Pitch Accent on a Content Word')