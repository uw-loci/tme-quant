import pandas as pd
from openpyxl import load_workbook
import os


def format_df_to_excel(
    df: pd.DataFrame, filename: str, sheet_name: str = "Sheet1", mode: str = "w"
):
    """
    Save a DataFrame to Excel and auto-adjust column widths for readability.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to save.
    filename : str
        Path to Excel file.
    sheet_name : str
        Name of the sheet to write to.
    mode : str
        'w' to create new file or overwrite, 'a' to append/update sheets.
    """
    # Determine if we need to append or create new
    if mode == "a" and os.path.exists(filename):
        with pd.ExcelWriter(
            filename, engine="openpyxl", mode="a", if_sheet_exists="replace"
        ) as writer:
            df.to_excel(writer, sheet_name=sheet_name, index=False)
    else:
        # Create new file
        df.to_excel(filename, sheet_name=sheet_name, index=False)

    # Load workbook and adjust column widths
    wb = load_workbook(filename)
    ws = wb[sheet_name]

    # Freeze the header row (row 1) so it stays visible when scrolling
    ws.freeze_panes = "A2"

    for col in ws.columns:
        max_length = 0
        col_letter = col[0].column_letter
        for cell in col:
            if cell.value is not None:
                max_length = max(max_length, len(str(cell.value)))
        ws.column_dimensions[col_letter].width = max_length + 2

    wb.save(filename)
