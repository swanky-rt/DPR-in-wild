import json

# --- your 5 queries (copied from your file) ---
queries = {
    "q1": "Can I get a list of athlete performance data for long-distance running events like 10,000 metres, 1500 metres, 3000 metres steeplechase, and 5000 metres, including their participation in European Team Championships Super League events, along with details on demographics, event locations, historical context, and metrics like times, rankings, and points to analyze trends and patterns in athletics?",

    "q2": "How can I get a list of cross-country athletes with their championship rankings, nationalities, and finishing times, along with their Olympic participation history, including the number of times they competed and their best positions, to analyze their career achievements and Olympic success?",

    "q3": "I need a list of movies from different years and countries, including details like title, director, studio, revenue, and rankings, as well as the people involved like writers, composers, and actors, so I can compare their financial performance and identify key trends and personnel.",

    "q4": "How can I get information on Olympic athletes' performances and trends, including their sports, medals, nationalities, and backgrounds, to analyze their achievements and demographics over time?",

    "q5": "I need a comprehensive dataset of Ohio General Assembly and Indiana House of Representatives members including districts party affiliations residences election histories term limits and demographic information about their areas of representation such as population founding year notable landmarks and historical context to analyze election trends and regional characteristics."
}

# --- generate expanded mapping ---
output = []

for i in range(1, 6):  # q1 to q5
    q_key = f"q{i}"
    for j in range(1, 21):  # 20 DPRs per query
        output.append({
            "dpr_id": f"{q_key}_{j}",
            "user_query": queries[q_key]
        })

# --- save file ---
with open("queries_expanded_for_stage4.json", "w") as f:
    json.dump(output, f, indent=2)

print("File generated: queries_expanded_for_stage4.json")