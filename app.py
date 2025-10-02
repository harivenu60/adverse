# file: app.py

from flask import Flask, request, render_template_string, flash
import requests, re, difflib
from datetime import datetime, timedelta
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk

nltk.download('vader_lexicon')

app = Flask(__name__)
app.secret_key = "change_this_secret"

# === API KEYS (replace with your own) ===
NEWSDATA_API_KEY = "YOUR_NEWSDATA_API_KEY"
NEWSAPI_API_KEY = "YOUR_NEWSAPI_API_KEY"
GNEWS_API_KEY = "YOUR_GNEWS_API_KEY"
WORLDNEWS_API_KEY = "YOUR_WORLDNEWS_KEY"
CURRENTS_API_KEY = "YOUR_CURRENTS_KEY"

# === API Endpoints ===
NEWSDATA_ENDPOINT = "https://newsdata.io/api/1/news"
NEWSAPI_ENDPOINT = "https://newsapi.org/v2/everything"
GNEWS_ENDPOINT = "https://gnews.io/api/v4/search"
WORLDNEWS_ENDPOINT = "https://api.worldnewsapi.com/search-news"
CURRENTS_ENDPOINT = "https://api.currentsapi.services/v1/search"

# Sanctions Lists
OFAC_URL = "https://sanctionslistservice.ofac.treas.gov/api/Publication"
EU_URL = "https://webgate.ec.europa.eu/fsd/fsf/public/files/xmlFullSanctionsList_1_1/content?token=dG9rZW4tMjAxNw"
UN_URL = "https://scsanctions.un.org/resources/xml/en/consolidated.xml"
UK_URL = "https://assets.publishing.service.gov.uk/government/uploads/system/uploads/attachment_data/file/1159796/UK_Sanctions_List.json"
OPENSANCTIONS_URL = "https://api.opensanctions.org/datasets/default/entities/"

# Initialize sentiment analyzer
sia = SentimentIntensityAnalyzer()

# ---------------------------
# Helpers
# ---------------------------
def highlight_terms(text, terms):
    for term in sorted(set(terms), key=lambda s: -len(s)):
        text = re.sub(re.escape(term), f"<mark>{term}</mark>", text, flags=re.IGNORECASE)
    return text

def extract_snippet(text, match_terms, radius=120):
    text_lower = text.lower()
    for term in match_terms:
        pos = text_lower.find(term.lower())
        if pos >= 0:
            start, end = max(0, pos-radius), min(len(text), pos+len(term)+radius)
            return ("..." if start>0 else "") + text[start:end] + ("..." if end<len(text) else "")
    return text[:radius*2] + "..."

def is_negative(text):
    if not text: return False
    score = sia.polarity_scores(text)
    return score["compound"] < -0.1  # negative threshold

# ---------------------------
# Sanctions Fetchers
# ---------------------------
def fetch_ofac_list():
    try:
        resp = requests.get(OFAC_URL, timeout=30).json()
        names = []
        for entry in resp.get("SDNList", {}).get("SDNEntries", []):
            if entry.get("lastName"): names.append(entry["lastName"])
            if entry.get("firstName"): names.append(entry["firstName"])
            for aka in entry.get("akaList", {}).get("aka", []):
                if aka.get("akaName"): names.append(aka["akaName"])
        return list(set(names))
    except: return []

def fetch_opensanctions():
    try:
        resp = requests.get(OPENSANCTIONS_URL, timeout=30).json()
        return [e.get("properties", {}).get("name") for e in resp.get("results", []) if e.get("properties", {}).get("name")]
    except: return []

def fetch_uk_list():
    try:
        resp = requests.get(UK_URL, timeout=30).json()
        return [x.get("Name") for x in resp if x.get("Name")]
    except: return []

# NOTE: EU + UN are XML — would need xml parser (skipped for brevity)
def search_sanctions(name_query):
    sanctioned = fetch_ofac_list() + fetch_opensanctions() + fetch_uk_list()
    sanctioned = [s for s in sanctioned if s]
    matches = []
    for sname in sanctioned:
        ratio = difflib.SequenceMatcher(None, name_query.lower(), sname.lower()).ratio()
        if ratio >= 0.8:
            matches.append({"sanctioned_name": sname, "similarity": round(ratio,2)})
    return sorted(matches, key=lambda x: -x["similarity"])

# ---------------------------
# News API wrappers
# ---------------------------
def fetch_from_newsdata(query, from_date, to_date):
    params = {"apikey": NEWSDATA_API_KEY, "q": query, "from_date": from_date, "to_date": to_date, "language": "en"}
    r = requests.get(NEWSDATA_ENDPOINT, params=params).json()
    return [{"title":a.get("title",""), "desc":a.get("description",""), "date":a.get("pubDate",""), "url":a.get("link",""), "source":"NewsData"} for a in r.get("results",[])]

def fetch_from_newsapi(query, from_date, to_date):
    params = {"apiKey": NEWSAPI_API_KEY, "q": query, "from": from_date, "to": to_date, "language": "en"}
    r = requests.get(NEWSAPI_ENDPOINT, params=params).json()
    return [{"title":a.get("title",""), "desc":a.get("description",""), "date":a.get("publishedAt",""), "url":a.get("url",""), "source":"NewsAPI"} for a in r.get("articles",[])]

def fetch_from_gnews(query, from_date, to_date):
    params = {"token": GNEWS_API_KEY, "q": query, "from": from_date, "to": to_date, "lang": "en"}
    r = requests.get(GNEWS_ENDPOINT, params=params).json()
    return [{"title":a.get("title",""), "desc":a.get("description",""), "date":a.get("publishedAt",""), "url":a.get("url",""), "source":"GNews"} for a in r.get("articles",[])]

def fetch_from_worldnews(query, from_date, to_date):
    params = {"api-key": WORLDNEWS_API_KEY, "text": query, "earliest-publish-date": from_date, "latest-publish-date": to_date, "language":"en"}
    r = requests.get(WORLDNEWS_ENDPOINT, params=params).json()
    return [{"title":a.get("title",""), "desc":a.get("summary",""), "date":a.get("publish_date",""), "url":a.get("url",""), "source":"WorldNews"} for a in r.get("news",[])]

def fetch_from_currents(query, from_date, to_date):
    params = {"apiKey": CURRENTS_API_KEY, "keywords": query, "start_date": from_date, "end_date": to_date, "language":"en"}
    r = requests.get(CURRENTS_ENDPOINT, params=params).json()
    return [{"title":a.get("title",""), "desc":a.get("description",""), "date":a.get("published",""), "url":a.get("url",""), "source":"Currents"} for a in r.get("news",[])]

# ---------------------------
# Unified search
# ---------------------------
def search_all(name, keywords, from_date, to_date):
    query = " ".join([name]+keywords if name else keywords)
    articles = []
    articles += fetch_from_newsdata(query, from_date, to_date)
    articles += fetch_from_newsapi(query, from_date, to_date)
    articles += fetch_from_gnews(query, from_date, to_date)
    articles += fetch_from_worldnews(query, from_date, to_date)
    articles += fetch_from_currents(query, from_date, to_date)

    results = []
    for art in articles:
        text = f"{art['title']} {art['desc']}"
        if not any(kw.lower() in text.lower() for kw in keywords): continue
        if not is_negative(text): continue  # sentiment filter
        snippet = extract_snippet(text, keywords+[name])
        results.append({
            "title": art["title"], "source": art["source"], "date": art["date"],
            "snippet": highlight_terms(snippet, keywords+[name]), "url": art["url"]
        })
    return results

# ---------------------------
# Flask UI
# ---------------------------
TEMPLATE = """
<!doctype html><html><body style="font-family:Arial;max-width:900px;margin:auto;">
<h2>Adverse News + Sanctions Search</h2>
<form method="post">
  Name: <input name="name" value="{{ request.form.name }}"><br>
  Keywords (comma): <input name="keywords" value="{{ request.form.keywords }}"><br>
  <button type="submit">Search</button>
</form>

{% if results %}
<h3>Negative News Matches ({{ results|length }})</h3>
{% for r in results %}
<div style="border:1px solid #ccc;padding:8px;margin:6px;">
  <b>{{ r.title }}</b> ({{ r.source }}, {{ r.date }})<br>
  {{ r.snippet|safe }}<br>
  <a href="{{ r.url }}" target="_blank">Read more</a>
</div>
{% endfor %}
{% endif %}

{% if ofac %}
<h3>Sanctions Matches ({{ ofac|length }})</h3>
<ul>
{% for o in ofac %}
  <li>{{ o.sanctioned_name }} (similarity: {{ o.similarity }})</li>
{% endfor %}
</ul>
{% endif %}
</body></html>
"""

@app.route("/", methods=["GET","POST"])
def index():
    results, ofac_results = None, None
    if request.method=="POST":
        name = request.form.get("name","").strip()
        kws = [k.strip() for k in request.form.get("keywords","").split(",") if k.strip()]
        to_date = datetime.today().strftime("%Y-%m-%d")
        from_date = (datetime.today()-timedelta(days=365*7)).strftime("%Y-%m-%d")
        try:
            results = search_all(name, kws, from_date, to_date)
            if name:
                ofac_results = search_sanctions(name)
        except Exception as e:
            flash(str(e))
    return render_template_string(TEMPLATE, results=results, ofac=ofac_results, request=request)

if __name__=="__main__":
    app.run(debug=True, port=5000)
