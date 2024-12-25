from flask import Flask, request, jsonify, session, render_template
from datetime import datetime, timedelta
from dateutil import parser
from groq import Groq
import os
import json
from flask_cors import CORS

app = Flask(__name__)
app.secret_key = "supersecretkey"  # Replace with a secure secret key

CORS(app)

class MetricQueryProcessor:
    def __init__(self, api_key: str, model: str = "llama3-8b-8192"):
        if not api_key:
            raise ValueError("API key cannot be None or empty")
        
        self.client = Groq(api_key=api_key)
        self.model = model

    def _get_default_dates(self) -> tuple[str, str]:
        end_date = datetime.now()
        start_date = end_date - timedelta(days=365)
        return start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d")

    def _normalize_date(self, date_str: str) -> str:
        try:
            parsed_date = parser.parse(date_str)
            return parsed_date.strftime("%Y-%m-%d")
        except (ValueError, TypeError):
            return date_str

    def _extract_info_with_llm(self, query: str, history: list) -> dict:
        system_prompt = """You are a helpful assistant that extracts company names, performance metrics, and dates from queries.
        Always return the response in valid JSON format with the following structure:
        {
            "entities": ["company1", "company2"],
            "parameter": "metric_name",
            "dates": {"start": "date1", "end": "date2"}
        }
        Only include dates if explicitly mentioned in the query."""
        
        user_prompt = f"""Query: {query}
    Previous context:
    {json.dumps(history[-5:], indent=2) if history else "No previous context"}

    Extract the relevant information and return it in the specified JSON format."""
        
        try:
            print("Sending the following prompt to LLM:")
            print(user_prompt)  # Add logging to inspect the prompt
            
            response = self.client.chat.completions.create(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                model=self.model,
            )
            print("Received response from LLM:")
            print(response)  # Add logging to inspect the response
            return json.loads(response.choices[0].message.content)
        except Exception as e:
            print(f"LLM error: {str(e)}")  # More detailed error logging
            raise Exception(f"LLM processing failed: {str(e)}")


    def process_query(self, query: str, history: list) -> list:
        extracted_info = self._extract_info_with_llm(query, history)
        default_start, default_end = self._get_default_dates()
        results = []
        for entity in extracted_info["entities"]:
            result = {
                "entity": entity,
                "parameter": extracted_info["parameter"],
                "start_date": self._normalize_date(
                    extracted_info.get("dates", {}).get("start", default_start)
                ),
                "end_date": self._normalize_date(
                    extracted_info.get("dates", {}).get("end", default_end)
                ),
            }
            results.append(result)
        return results

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/process_query', methods=['POST'])
def process_query():
    data = request.get_json()
    query = data.get("query", "").strip()

    if not query:
        return jsonify({"error": "Query cannot be empty"}), 400

    # Initialize session history if not already present
    if "history" not in session:
        session["history"] = []

    processor = MetricQueryProcessor(api_key=os.getenv("GROQ_API_KEY"))

    try:
        # Log current session history for debugging
        print("Current session history:")
        print(session["history"])

        # Process query and get results
        results = processor.process_query(query, session["history"])
        
        # Maintain session history and append the current query and results
        interaction = {"query": query, "results": results}
        session["history"].append(interaction)

        # Limit history to the last 6 queries to avoid storing too much data
        session["history"] = session["history"][-6:]

        # Log the results for debugging
        print("Results from LLM:")
        print(results)

        # Return only the results of the current query
        return jsonify(results)
    except Exception as e:
        print(f"Error processing query: {e}")
        return jsonify({"error": str(e)}), 500



@app.route('/clear_history', methods=['POST'])
def clear_history():
    session.pop("history", None)
    return jsonify({"message": "History cleared."}), 200

@app.route('/end_session', methods=['POST'])
def end_session():
    session.clear()
    return jsonify({"message": "Session ended."}), 200

if __name__ == '__main__':
    app.run(debug=True)
