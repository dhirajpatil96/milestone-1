import pandas as pd
from bs4 import BeautifulSoup
import re
import email
from email import policy
from email.parser import BytesParser
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize


try:
       df = pd.read_csv('raw_emails.csv')
       print(f"Loaded {len(df)} raw emails from CSV.")
except FileNotFoundError:
       print("Error: raw_emails.csv not found. Run Step 1 first.")
       exit()

if df.empty:
       print("Error: raw_emails.csv is empty. Check Step 1.")
       exit()

def parse_email(raw_email_str):
       try:
           msg = BytesParser(policy=policy.default).parsebytes(raw_email_str.encode('utf-8'))
           subject = msg.get('Subject', '')
           body = ''
           if msg.is_multipart():
               for part in msg.walk():
                   if part.get_content_type() == 'text/plain':
                       body = part.get_payload(decode=True).decode('utf-8', errors='ignore')
                       break
           else:
               body = msg.get_payload(decode=True).decode('utf-8', errors='ignore')
           return subject, body
       except Exception as e:
           print(f"Error parsing email: {e}")
           return '', ''

parsed_results = df['raw_email'].apply(parse_email)
print(f"Parsed results length: {len(parsed_results)}")
if len(parsed_results) == 0:
       print("No emails parsed. Check raw_email column.")
       exit()

try:
       df['subject'], df['cleaned_body'] = zip(*parsed_results)
except ValueError as e:
       print(f"Unpacking error: {e}. Parsed results: {parsed_results.head()}")
       exit()

def clean_text(text):
       soup = BeautifulSoup(text, 'html.parser')
       text = soup.get_text()
       text = re.sub(r'(--\n.*|Best regards.*|Sent from.*|Regards,.*)', '', text, flags=re.MULTILINE | re.IGNORECASE)
       text = re.sub(r'\s+', ' ', text).strip().lower()
       return text

stop_words = set(stopwords.words('english'))
def remove_stopwords(text):
       try:
           words = word_tokenize(text)
           return ' '.join([w for w in words if w not in stop_words and w.isalnum()])
       except Exception as e:
           print(f"Tokenization error: {e}. Returning original text.")
           return text

df['cleaned_body'] = df['cleaned_body'].apply(clean_text)
df['subject'] = df['subject'].apply(clean_text)

df['full_text'] = df['subject'] + ' ' + df['cleaned_body']

df['cleaned_text'] = df['full_text'].apply(remove_stopwords)

df.drop_duplicates(subset='cleaned_text', inplace=True)

df = df[['subject', 'cleaned_body', 'full_text', 'cleaned_text']]
df.to_csv('cleaned_emails.csv', index=False)

print("Cleaned emails saved to cleaned_emails.csv")
print(f"Total emails after cleaning: {len(df)}")
   