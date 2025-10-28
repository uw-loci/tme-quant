import pandas as pd
from openpyxl import load_workbook


def format_df_to_excel(df: pd.DataFrame, filename: str, sheet_name: str = "Sheet1"):
    """Save a DataFrame to Excel and auto-adjust column widths for readability."""
    # Save DataFrame to Excel
    df.to_excel(filename, sheet_name=sheet_name, index=False)

    # Load workbook and get sheet
    wb = load_workbook(filename)
    ws = wb[sheet_name]

    # Adjust column widths
    for col in ws.columns:
        max_length = 0
        col_letter = col[0].column_letter
        for cell in col:
            if cell.value is not None:
                max_length = max(max_length, len(str(cell.value)))
        ws.column_dimensions[col_letter].width = max_length + 2  # add padding

    # Save workbook
    wb.save(filename)
