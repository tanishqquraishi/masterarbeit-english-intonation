import pandas as pd
from pymer4.models import Lmer
from utils import load_data, rename_columns, create_binary_column
from visualizations_binomial_models import plot_coefficients, plot_likelihood_by_group, plot_likelihood_by_interaction

# Load data
file_path = r"C:\\Users\\Tanishq\\Documents\\stuttgart\\Study\\thesis\\data\\model data\\pa_con_17.07.2024.xlsx"
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
plot_coefficients(model_df, title='GLMM Coefficients for Content Word PA')
plot_likelihood_by_group(data, 'bilingual', 'Likelihood of Pitch Accent by Speaker Group', 'Likelihood of Pitch Accent (%)')
plot_likelihood_by_group(data, 'formality', 'Likelihood of Pitch Accent by Formality', 'Likelihood of Pitch Accent (%)')