
import pypdf
import sys
from pathlib import Path

def extract_text_to_file(pdf_path, output_path):
    try:
        reader = pypdf.PdfReader(pdf_path)
        with open(output_path, "w", encoding="utf-8") as f:
            for i, page in enumerate(reader.pages):
                f.write(f"--- Page {i+1} ---\n")
                f.write(page.extract_text() + "\n")
        print(f"Text saved to {output_path}")
    except Exception as e:
        print(str(e))

if __name__ == "__main__":
    pdf_path = r"c:\Users\Siddhant\AI Exp\PBL-1 Final Report2 (1).pdf"
    output_path = r"c:\Users\Siddhant\AI Exp\isl-translator\scripts\pdf_content.txt"
    extract_text_to_file(pdf_path, output_path)
