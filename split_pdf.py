import pypdf
import os

def split_pdf(pdf_path, output_dir, pages_per_file=5):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    with open(pdf_path, 'rb') as f:
        reader = pypdf.PdfReader(f)
        total_pages = len(reader.pages)
        print(f"Total pages in PDF: {total_pages}")

        for i in range(0, total_pages, pages_per_file):
            end_page = min(i + pages_per_file, total_pages)
            start_page = i
            
            content = ""
            for page_num in range(start_page, end_page):
                page = reader.pages[page_num]
                content += page.extract_text() + "\n\n"

            file_name = f"pages_{start_page+1}_to_{end_page}.txt"
            file_path = os.path.join(output_dir, file_name)
            
            with open(file_path, 'w', encoding='utf-8') as out_f:
                out_f.write(content)
            
            print(f"Created: {file_name}")

if __name__ == "__main__":
    pdf_filename = "_OceanofPDF.com_Dungeons_and_Dragons_Players_Handbook_2024_-_Wizards_of_the_Coast.pdf"
    output_directory = "split_pdf_content"
    split_pdf(pdf_filename, output_directory)
