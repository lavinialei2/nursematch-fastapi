from fastapi import FastAPI
from pydantic import BaseModel
from typing import Any

import pandas as pd
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

#nltk.download('all')

import spacy

# extended keywords list by ChatGPT
categories = {
    'pay': ['pay', 'salary', 'compensation', 'wages', 'hourly rate', 'overtime', 'paycheck', 'bonuses', 'stipend', 'financial package', 'incentives', 'earnings', 'pay scale', 'rate of pay', 'benefits package', 'taxable income', 'untaxed stipends', 'travel reimbursement', 'sign-on bonus', 'shift differentials', 'per diem pay'],
    'orientation & onboarding': ['orientation', 'onboarding', 'training', 'initial training', 'preparation', 'first day', 'introductory period', 'hospital orientation', 'onboarding process', 'facility training', 'new hire orientation', 'training schedule', 'competency', 'job shadowing', 'mentor program', 'skills assessment', 'preceptor', 'unit orientation', 'workflow training', 'EMR training', 'education program', 'skills check-off'],
    'mgmt & leadership': ['management', 'leadership', 'supervisor', 'manager', 'charge nurse', 'head nurse', 'team lead', 'nursing administration', 'leadership style', 'supportive management', 'unsupportive management', 'communication', 'transparency', 'accountability', 'upper management', 'nurse manager', 'leadership team', 'director of nursing', 'unit manager', 'staff relations', 'workplace culture', 'decision-making', 'conflict resolution', 'management policies', 'administration', 'bureaucracy', 'hierarchy', 'managerial support'],
    'safety & patient ratios': ['safety', 'patient ratios', 'workload', 'staffing levels', 'understaffing', 'overworked', 'patient load', 'nurse-to-patient ratio', 'staffing shortage', 'safe staffing', 'unsafe conditions', 'workplace safety', 'patient care', 'acuity', 'burnout', 'workload intensity', 'assigned patients', 'adequate staffing', 'job stress', 'breaks', 'mandatory overtime', 'safety protocols', 'overwhelmed', 'patient assignments', 'workplace fatigue', 'workplace hazards', 'shift workload', 'nursing safety', 'work-life balance', 'emergency situations', 'staffing flexibility', 'triage', 'float pool'],
    'DEI/LGBTQ+ friendliness': ['DEI', 'diversity', 'inclusion', 'LGBTQ+', 'equity', 'workplace diversity', 'LGBTQ+ friendly', 'inclusive environment', 'equality', 'bias', 'discrimination', 'unconscious bias', 'cultural sensitivity', 'gender inclusion', 'race and ethnicity', 'equal opportunity', 'multicultural', 'gender identity', 'sexual orientation', 'BIPOC', 'inclusive language', 'workplace belonging', 'minority-friendly', 'safe space', 'gender expression', 'inclusive healthcare', 'respect', 'prejudice', 'intersectionality', 'equal treatment', 'allyship', 'cultural competence', 'underrepresented', 'affirmative action', 'inclusive policies'],
    'patient acuity': ['patient acuity', 'acuity', 'severity', 'critical patients', 'high acuity', 'low acuity', 'severity of illness', 'acuity levels', 'intensive care', 'emergency situations', 'high-intensity patients', 'complex cases', 'critical care', 'case complexity', 'medical conditions', 'ICU', 'ER', 'trauma', 'serious conditions', 'chronic illness', 'patient classification', 'care intensity', 'patient deterioration', 'rapid response', 'emergent care', 'severity assessment', 'triage', 'acuity index', 'urgent care', 'stable vs. critical', 'clinical severity'],
    'housing options': ['housing', 'accommodation', 'living options', 'housing stipend', 'housing allowance', 'provided housing', 'on-site housing', 'travel nurse housing', 'temporary housing', 'relocation package', 'living arrangements', 'off-campus housing', 'housing proximity', 'affordable housing', 'housing conditions', 'living stipend', 'housing benefits', 'housing quality', 'safe accommodation', 'furnished housing', 'short-term housing', 'rental options', 'extended stay', 'hotel accommodations', 'travel accommodations', 'airbnb', 'housing availability', 'local housing', 'housing support', 'housing expenses'],
    'facility location': ['facility location', 'location', 'proximity', 'hospital location', 'commute', 'nearby', 'city', 'urban', 'rural', 'suburban', 'proximity to housing', 'transportation options', 'traffic', 'parking', 'public transit', 'location convenience', 'nearby amenities', 'neighborhood', 'accessible location', 'geographic area', 'distance', 'regional', 'local', 'nearby attractions', 'neighborhood safety', 'commute time', 'walkability', 'location safety', 'close to home', 'isolated', 'metropolitan area']
}

# Extract sentences based on keywords
def extract_relevant_sentences(sentences, categories):
    category_sentences = {category: [] for category in categories}

    for sentence in sentences:
        for category, keywords in categories.items():
            if any(keyword in sentence.lower() for keyword in keywords):
                category_sentences[category].append(sentence)

    return category_sentences



def analyze_sentiment(sentences):

    analyzer = SentimentIntensityAnalyzer()
    sentiments = []
    for sentence in sentences:
        sscore = analyzer.polarity_scores(sentence)
        sentiments.append(sscore['compound'])
    return sentiments


def adjust_scores(scores, category_sentiments):
    adjusted_scores = scores.copy()

    for category, sentiments in category_sentiments.items():
        if sentiments:
            avg_sentiment = sum(sentiments) / len(sentiments)
            avg_sentiment = (avg_sentiment+1)*2.5

            adjusted_scores[category] = round((float(adjusted_scores[category]) + avg_sentiment)/2, 2) # Equal weighting between given scores and NLP
            adjusted_scores[category] = max(1, min(adjusted_scores[category], 5))

    return adjusted_scores

# Extract sentences based on keywords
def extract_relevant_sentences(sentences, categories):
    category_sentences = {category: [] for category in categories}

    for sentence in sentences:
        for category, keywords in categories.items():
            if any(keyword in sentence.lower() for keyword in keywords):
                category_sentences[category].append(sentence)

    return category_sentences

def NLPAdjust(review, scores, printOn=False):
    # Load spaCy model
    nlp = spacy.load('en_core_web_sm')
    # Tokenize the review into sentences
    sentences = [sent.text for sent in nlp(review).sents]
    category_sentences = extract_relevant_sentences(sentences, categories)
    category_sentiments = {category: analyze_sentiment(sentences) for category, sentences in category_sentences.items()}
    adjusted_scores = adjust_scores(scores, category_sentiments)

    if printOn:
      print(category_sentences)
      print(category_sentiments)
      print("Original Scores: ", scores)
      print("Adjusted Scores: ", adjusted_scores)

    return adjusted_scores


def hospitalReviews (df):
  reviews_by_hospital = df.groupby('Hospital')

  for hospital, reviews in reviews_by_hospital:
      print(f"Hospital: {hospital}")
      for index, row in reviews.iterrows():
          scores = {
              'pay': row['Pay'],
              'orientation & onboarding': row['Orientation & Onboarding'],
              'mgmt & leadership': row['Mgmt & Leadership'],
              'safety & patient ratios': row['Safety & Patient Ratios'],
              'DEI/LGBTQ+ friendliness': row['DEI/LGBTQ+ Friendliness'],
              'patient acuity': row['Patient Acuity'],
              'housing options': row['Housing Options'],
              'facility location': row['Facility Location']
          }
          review = row['Review']
          adjusted_scores = NLPAdjust(review, scores)

          df.loc[index, ['Pay', 'Orientation & Onboarding', 'Mgmt & Leadership',
                          'Safety & Patient Ratios', 'DEI/LGBTQ+ Friendliness',
                          'Patient Acuity', 'Housing Options', 'Facility Location']] = list(adjusted_scores.values())


# Initialize FastAPI app
app = FastAPI()

# Define input and output models
class InputModel(BaseModel):
    param1: str

class OutputModel(BaseModel):
    result: Any

def dfToJson(df: pd.DataFrame) -> dict:
    """
    Converts a DataFrame to a JSON-compatible dictionary.
    Includes types to preserve fidelity (e.g., for dates).
    """
    return df.to_dict(orient='records')

def jsonToDf(data: list[dict]) -> pd.DataFrame:
    """
    Converts a JSON-compatible list of dictionaries back to a DataFrame.
    """
    return pd.DataFrame(data)


class DataFrameInput(BaseModel):
    data: list[dict]


# Define your main driver function
def score_adjustment_driver(param1: str):
    df_url = param1
    df = pd.read_csv(df_url)
    df = df.rename(columns={
    'Pay (1-5)': 'Pay',
    'Orientation & onboarding (1-5)': 'Orientation & Onboarding',
    'Mgmt & leadership (1-5)': 'Mgmt & Leadership',
    'Safety & patient ratios (1-5)': 'Safety & Patient Ratios',
    'DEI/LGBTQ+ friendliness (1-5)': 'DEI/LGBTQ+ Friendliness',
    'Patient Acuity (1-5)': 'Patient Acuity',
    'Housing options (1-5)': 'Housing Options',
    'Facility Location (1-5)': 'Facility Location',
    'Would Return Again Y/N' : 'Would Return Again'
    })

    hospitalReviews(df)

    return df


def add_dummy_reviews(df):
    hospitals = df['Hospital'].unique()

    for hospital in hospitals:
        dummy_nurse5 = {
        'Nurse': 'D5',
        'Hospital': hospital,
        'Review': '',
        'Pay': 5,
        'Orientation & Onboarding': 5,
        'Mgmt & Leadership' : 5,
        'Safety & Patient Ratios': 5,
        'DEI/LGBTQ+ Friendliness': 5,
        'Patient Acuity': 5,
        'Housing Options': 5,
        'Facility Location': 5,
        'Would Return Again': 'Yes'
        }
        d5 = pd.DataFrame([dummy_nurse5])
        df = pd.concat([df, d5], ignore_index=True)

        dummy_nurse1 = {
        'Nurse': 'D1',
        'Hospital': hospital,
        'Review': '',
        'Pay': 1,
        'Orientation & Onboarding': 1,
        'Mgmt & Leadership' : 1,
        'Safety & Patient Ratios': 1,
        'DEI/LGBTQ+ Friendliness': 1,
        'Patient Acuity': 1,
        'Housing Options': 1,
        'Facility Location': 1,
        'Would Return Again': 'Yes'
        }
        d1 = pd.DataFrame([dummy_nurse1])
        df = pd.concat([df, d1], ignore_index=True)

def averaging_driver(param_1):
    df_url = param_1
    df = pd.read_csv(df_url)
    add_dummy_reviews(df)
    df['Facility Location'] = df['Facility Location'].astype(float)

    score_category = ['Pay', 'Orientation & Onboarding', 'Mgmt & Leadership', 'Safety & Patient Ratios', 'DEI/LGBTQ+ Friendliness', 'Patient Acuity', 'Housing Options', 'Facility Location']
    grouped = df.groupby('Hospital')[score_category].mean()
    return grouped


def leaderboard_driver(param1):
    df_url = param1
    df = pd.read_csv(df_url)

    score_category = ['Pay', 'Orientation & Onboarding', 'Mgmt & Leadership', 'Safety & Patient Ratios', 'DEI/LGBTQ+ Friendliness', 'Patient Acuity', 'Housing Options', 'Facility Location']
    
    for category in score_category:
        print(f"Top 5 Hospitals in {category}:")
        top_5 = df.nlargest(5, category)
        print(top_5.reset_index()[['Hospital', category]])

        print(f"\nBottom 5 Hospitals in {category}:")
        bottom_5 = df.nsmallest(5, category)
        print(bottom_5.reset_index()[['Hospital', category]])
        print("\n", "=" * 50, "\n")

    return "Leaderboard printed."


def full_driver(param1):
    df_url = param1
    df = pd.read_csv(df_url)

    df = df.rename(columns={
    'Pay (1-5)': 'Pay',
    'Orientation & onboarding (1-5)': 'Orientation & Onboarding',
    'Mgmt & leadership (1-5)': 'Mgmt & Leadership',
    'Safety & patient ratios (1-5)': 'Safety & Patient Ratios',
    'DEI/LGBTQ+ friendliness (1-5)': 'DEI/LGBTQ+ Friendliness',
    'Patient Acuity (1-5)': 'Patient Acuity',
    'Housing options (1-5)': 'Housing Options',
    'Facility Location (1-5)': 'Facility Location',
    'Would Return Again Y/N' : 'Would Return Again'
    })

    hospitalReviews(df)

    add_dummy_reviews(df)
    df['Facility Location'] = df['Facility Location'].astype(float)

    score_category = ['Pay', 'Orientation & Onboarding', 'Mgmt & Leadership', 'Safety & Patient Ratios', 'DEI/LGBTQ+ Friendliness', 'Patient Acuity', 'Housing Options', 'Facility Location']
    grouped = df.groupby('Hospital')[score_category].mean()

    for category in score_category:
        print(f"Top 5 Hospitals in {category}:")
        top_5 = grouped.nlargest(5, category)
        print(top_5.reset_index()[['Hospital', category]])

        print(f"\nBottom 5 Hospitals in {category}:")
        bottom_5 = grouped.nsmallest(5, category)
        print(bottom_5.reset_index()[['Hospital', category]])
        print("\n", "=" * 50, "\n")

    return grouped


# API route for score adjustment function
@app.post("/adjust-scores/", response_model=OutputModel)
async def run_score_adjustment_driver(input_data: InputModel):
    result = score_adjustment_driver(input_data.param1)
    return dfToJson(result)

# API route for hospital averaging function
@app.post("/average-hospital-scores/", response_model=OutputModel)
async def run_averaging_driver(input_data: InputModel):
    result = averaging_driver(input_data.param1)
    return dfToJson(result)


# API route for printing leaderboard
@app.post("/hospital-leaderboard/", response_model=OutputModel)
async def run_leaderboard_driver(input_data: InputModel):
    result = leaderboard_driver(input_data.param1)
    return {"result": result}

# API route for full processing
@app.post("/full-score-processing/", response_model=OutputModel)
async def run_full_driver(input_data: InputModel):
    result = full_driver(input_data.param1)
    return dfToJson(result)
