import os
import pandas as pd
from openai import AzureOpenAI
from dotenv import load_dotenv


# Load environment variables

load_dotenv()

# Config
api_version = "2024-12-01-preview"
subscription_key = os.getenv("AZURE_OPENAI_KEY")
endpoint = "https://openai-poc-dev-uksouth.openai.azure.com/"
model_name = "gpt-4o-mini"
deployment = "gpt-4o-mini"

client = AzureOpenAI(
    api_version=api_version,
    azure_endpoint=endpoint,
    api_key=subscription_key,
)

# Load Lancaster CSV
df = pd.read_csv(r"Lancaster_Sensorimotor_40k_database.csv") 
# https://embodiedcognitionlab.shinyapps.io/sensorimotor_norms/
# Lynott, D., Connell, L., Brysbaert, M., Brand, J., & Carney, J. (2019). The Lancaster Sensorimotor Norms: Multidimensional measures of perceptual and action strength for 40,000 English words.


# Define missing words

missing_words = [
    " fill in your words that need imputation"
]


#  Build GPT prompt
prompt = f"""
You are a cognitive linguist familiar with the Lancaster Sensorimotor Norms dataset (Vankrunkelsven et al., 2019). 
This dataset provides human ratings of how concepts are experienced in different sensory and motor modalities.

Columns in the dataset include:
- Word
- Auditory.mean: mean rating (0-5) of how strongly the concept is experienced by hearing
- Haptic.mean: mean rating (0-5) of how strongly the concept is experienced by touch
- Visual.mean: mean rating (0-5) of how strongly the concept is experienced by seeing
- Minkowski3.sensorimotor: an aggregated sensorimotor strength across 11 dimensions, calculated as the Minkowski distance with exponent 3 of the 11-dimensional vector from the origin

The 11 subdimensions used to compute Minkowski3 are:
1. Hand_arm.mean: strength of experience from hand/arm actions
2. Head.mean: strength of experience from head actions excluding mouth
3. Mouth.mean: strength of experience from mouth/throat actions
4. Torso.mean: strength of experience from torso actions
5. Gustatory.mean: strength of experience from tasting
6. Haptic.mean: strength of experience from feeling through touch
7. Interoceptive.mean: strength of experience from internal body sensations
8. Olfactory.mean: strength of experience from smelling
9. Visual.mean: strength of experience from seeing
10. Foot_leg.mean: strength of experience from foot/leg actions
11. Auditory.mean: strength of experience from hearing

For each of these words: {', '.join(missing_words)}, estimate the following using your best approximation based on how a human would perceive them:

- Auditory.mean
- Haptic.mean
- Visual.mean
- Minkowski3.sensorimotor

Use numeric ratings from 0-5.  
Output your results as a table with columns:

Word | Predicted_Auditory | Predicted_Haptic | Predicted_Visual | Predicted_Minkowski3

Do not include words not listed above.  
When estimating Minkowski3.sensorimotor, compute it as the Minkowski distance (exponent 3) of the 11-dimension vector above, using your predicted ratings.
Treat proper nouns like “St. Louis” or “Thomas Edison” as humans would: they may be low in Haptic or Auditory but could have moderate Visual ratings due to semantic associations.
Use your knowledge of the Lancaster norms and human sensory ratings as guidance.
"""


#  Call GPT
response = client.chat.completions.create(
    model=model_name,
    messages=[
        {"role": "system", "content": "You are an expert cognitive linguist."},
        {"role": "user", "content": prompt}
    ],
    temperature=0.0,
    max_tokens=1500,
)


#  Print results and token usage-
content = response.choices[0].message.content
prompt_tokens = response.usage.prompt_tokens
completion_tokens = response.usage.completion_tokens
total_tokens = response.usage.total_tokens

print("\n=== Sensorimotor Ratings Predictions ===\n")
print(content)
print("\n--- Token Usage ---")
print(f"Prompt tokens: {prompt_tokens}")
print(f"Completion tokens: {completion_tokens}")
print(f"Total tokens: {total_tokens}")
