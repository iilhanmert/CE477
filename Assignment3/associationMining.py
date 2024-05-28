import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules


data = pd.read_csv('diabetes_prediction_dataset.csv')

data['smoking_history'] = data['smoking_history'].replace('No Info', 'Unknown')

data_encoded = pd.get_dummies(data, columns=['gender', 'smoking_history'])

data_sampled = data_encoded.sample(n=1000, random_state=0)

# Convert categorical variables to boolean 
for col in data_sampled.columns:
    data_sampled[col] = data_sampled[col].astype(bool)

# Apriori algorithm
frequent_itemsets = apriori(data_sampled, min_support=0.01, use_colnames=True)

rules = association_rules(frequent_itemsets, metric='lift', min_threshold=1)

# Report association rules
interesting_rules = rules[(rules['lift'] > 1) & (rules['confidence'] > 0.5)]
print("Interesting Association Rules:")
print(interesting_rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']])

# Save the CSV file
interesting_rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']].to_csv('interesting_association_rules.csv', index=False)

print("Interesting association rules saved to 'interesting_association_rules.csv'")
