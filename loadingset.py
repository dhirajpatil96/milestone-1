import os
import email
from email import policy
from email.parser import BytesParser
import pandas as pd

def load_emails_from_dir(directory):
       emails = []
       total_files = 0
       for root, dirs, files in os.walk(directory):
           for file in files:
               total_files += 1
               try:
                   with open(os.path.join(root, file), 'rb') as f:
                       msg = BytesParser(policy=policy.default).parse(f)
                       emails.append(msg)
               except Exception as e:
                   print(f"Skipped {file}: {e}")
       print(f"Total files scanned: {total_files}, Emails loaded: {len(emails)}")
       return emails

directory = 'D:/email'
enron_emails = load_emails_from_dir(directory)

if len(enron_emails) == 0:
       print("No emails found. Check the directory path and file extensions.")
       exit()

df = pd.DataFrame({'raw_email': [str(email) for email in enron_emails]})
df.to_csv('raw_emails.csv', index=False)
print("Raw emails saved to raw_emails.csv")
   