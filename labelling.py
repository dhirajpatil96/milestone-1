import pandas as pd
import re

df = pd.read_csv('cleaned_emails.csv')

def rule_based_label(text):
    text = str(text).lower()
    if 'urgent' in text or 'asap' in text or 'deadline' in text:
        urgency = 'high'
    elif 'meeting' in text or 'project' in text:
        urgency = 'medium'
    else:
        urgency = 'low'
    
    if 'buy now' in text or 'discount' in text or 'unsubscribe' in text:
        category = 'promotional'
    elif 'meeting' in text or 'report' in text or 'work' in text:
        category = 'work'
    elif 'spam' in text or 'viagra' in text:
        category = 'spam'
    else:
        category = 'personal'
    
    return category, urgency

df['category'], df['urgency'] = zip(*df['cleaned_text'].apply(rule_based_label))

df.to_csv('labeled_emails.csv', index=False)
print("Labeled emails saved to labeled_emails.csv")
print(f"Total labeled emails: {len(df)}")
print(df[['category', 'urgency']].value_counts())
