import requests
import json
import pandas as pd

# ====================== CONFIG ======================
CUBE_URL = "http://192.168.1.4:4000/cubejs-api/v1/load"
HEADERS = {
    "Authorization": "Bearer supersecretkey",
    "Content-Type": "application/json"
}

# Weekly aggregation on loginDate (perfect for login-related features + future disbursal prediction)
# Target = totalDisbursedAmount (what we want to predict)
# Features include logins, amounts, ratios, scores, TATs — exactly what you asked for
query_payload = {
    "query": {
        "measures": [
            "Applications.loginCount",
            "Applications.avgSanctionAmount",
            "Applications.sanctionRatio",           # pre-computed ratio
            "Applications.totalDisbursedAmount",    # ← TARGET for prediction
        ],
        "timeDimensions": [
            {
                "dimension": "Applications.loginDate",
                "granularity": "week",
                "dateRange": ["2023-01-01", "2026-03-31"]   # ← Change this if you need a different period
            }
        ],
        "order": {
            "Applications.loginDate": "asc"
        },
        "limit": 1000
    }
}

# ====================== FETCH ======================
response = requests.post(CUBE_URL, headers=HEADERS, json=query_payload)

if response.status_code != 200:
    print("❌ Error fetching data:")
    print(response.status_code)
    print(response.text)
else:
    result = response.json()
    raw_data = result.get("data", [])

    if not raw_data:
        print("⚠️ No data returned. Try widening the dateRange or check if you have login data.")
    else:
        # Convert to clean pandas DataFrame
        df = pd.DataFrame(raw_data)
        
        # Remove "Applications." prefix and fix the week column name
        df.columns = [col.replace("Applications.", "") for col in df.columns]
        if "loginDate.week" in df.columns:
            df = df.rename(columns={"loginDate.week": "week"})
        
        # Optional: make week a proper datetime for easier time-series work later
        df["week"] = pd.to_datetime(df["week"])

        print("✅ Successfully fetched weekly aggregated data!")
        print(f"Shape: {df.shape[0]} weeks × {df.shape[1]} columns")
        print("\nPreview:")
        print(df.head(10))

        # Save exactly as you requested
        df.to_json("applications_data.json", orient="records", indent=4, date_format="iso")
        
        # Also save as CSV for easy viewing in Excel/VS Code
        df.to_csv("applications_data.csv", index=False)
        
        print('\n🎉 Files saved:')
        print('   → applications_data.json  (ready for ML)')
        print('   → applications_data.csv   (easy to inspect)')