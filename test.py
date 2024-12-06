import requests

# url = "http://127.0.0.1:8000/adjust-scores/"
# payload = {"param1": "https://docs.google.com/spreadsheets/d/1xAmGGrc1e341alKfzt3DR3yoBgY8vArx9wFPYsoCTyo/export?format=csv"}

# response = requests.post(url, json=payload)


# url = "http://127.0.0.1:8000/average-hospital-scores/"
# payload = {"param1": "https://docs.google.com/spreadsheets/d/e/2PACX-1vSWVe2_Onswyth6K4P5MwP0dDKmDjtSaGrQmJwHXL40x7TBhTqcL0IT4Q7MIWdDcF5wuAZ6sgZexPeV/pub?output=csv"}

# response = requests.post(url, json=payload)


# url = "http://127.0.0.1:8000/hospital-leaderboard/"
# payload = {"param1": "https://docs.google.com/spreadsheets/d/e/2PACX-1vSAFUKRdaxvPmCYFWLRH3mPLsupeEGMeGYofKaNi3QZSHtpuHk9H4PhXBEX3nE4DxvqTAHeuPVlVJQI/pub?gid=1585847606&single=true&output=csv"}

# response = requests.post(url, json=payload)

url = "http://127.0.0.1:8000/full-score-processing/"
payload = {"param1": "https://docs.google.com/spreadsheets/d/1xAmGGrc1e341alKfzt3DR3yoBgY8vArx9wFPYsoCTyo/export?format=csv"}

response = requests.post(url, json=payload)