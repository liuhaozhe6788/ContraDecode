import pandas as pd
import requests
from io import BytesIO
import argparse

def main():
    parser = argparse.ArgumentParser(description="Download and convert Google Sheets Excel to CSV.")
    parser.add_argument("--url", type=str, required=True, help="Direct export link to Google Sheets in .xlsx format")
    parser.add_argument("--language_pair", type=str, required=True, help="Specify the sheet from the link to download")
    parser.add_argument("--output", type=str, default="customized_dataset.csv", help="Output CSV file name")
    args = parser.parse_args()

    # Fetch the Excel file
    response = requests.get(args.url)
    response.raise_for_status()  # raises error if request failed
    xls_data = BytesIO(response.content)

    # Load and convert to CSV
    df = pd.read_excel(xls_data, sheet_name = args.language_pair)
    print(df.head(5))
    df.to_csv(f'customized_datasets/{args.output}', index=False)

    print(f"✅ Dataset saved to customized_datasets/{args.output}")

if __name__ == "__main__":
    main()
