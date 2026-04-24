## Kostråd-chattbot

En RAG-chattbot som ger kostråd för gravida, spädbarn och småbarn (0–2 år), 
baserad på information från Livsmedelsverket och 1177.

## Installation

1. Klona repot
2. Installera beroenden:
   pip install -r requirements.txt

3. Skapa en .env-fil i samma mapp med följande innehåll:
   GENAI_API_KEY=din_nyckel_här

   API-nyckeln får du på: https://aistudio.google.com/apikey

4. Kör appen:
   streamlit run livsmedelsverket_kostrad_chattbot.py

## Notering
Om embeddings.pkl inte finns skapas den automatiskt vid första uppstart.
Det tar några minuter beroende på API-hastighet.
