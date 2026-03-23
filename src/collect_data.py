import os
import requests
from bs4 import BeautifulSoup
from pdfminer.high_level import extract_text
from tqdm import tqdm
from langdetect import detect

os.makedirs("datasets/pdf", exist_ok = True)
os.makedirs("datasets/html", exist_ok= True)

pdf_links = {
    "pg_regulations": "https://iitj.ac.in/PageImages/Gallery/07-2025/Regulation_PG_2022-onwards_20022023.pdf",
    "btech_regulations": "https://iitj.ac.in/PageImages/Gallery/04-2025/BTech_Old_REV_13122014.pdf",
    "mtech_regulations": "https://iitj.ac.in/PageImages/Gallery/04-2025/MTech_New_REVISED_13Feb2017.pdf",
    "phd_regulations": "https://iitj.ac.in/PageImages/Gallery/06-2025/Ph.D._New.pdf",
    "bs_physics_curriculum": "https://iitj.ac.in/PageImages/Gallery/02-2025/Curriculum1-638755740799615537.pdf",
    "cse_course_details": "https://iitj.ac.in/PageImages/Gallery/07-2025/CSE-Courses-Details.pdf",
    "phd_curriculum": "https://www.iitj.ac.in/PageImages/Gallery/02-2025/curriculum-of-PhD-MT-Program--638756575643402782.pdf",
    "metallurgy_courses": "https://iitj.ac.in/PageImages/Gallery/02-2025/Courses2-638755737688449867.pdf",
    "research_projects": "https://iitj.ac.in/PageImages/Gallery/02-2025/Website-Research-Projects-638741930489723315.pdf",
    "annual_report_2023": "https://www.iitj.ac.in/PageImages/Gallery/03-2025/IITJ_AR_2022_2023_English.pdf"
}

html_links = {
    "iitj_homepage": "https://www.iitj.ac.in/",
    "faculty_directory": "https://www.iitj.ac.in/main/en/faculty-members",
    "academics_office": "https://www.iitj.ac.in/office-of-academics/en/academics",
    "chemical_faculty": "https://www.iitj.ac.in/chemical-engineering/en/faculty-members",
    "electrical_faculty": "https://www.iitj.ac.in/electrical-engineering/en/faculty-members",
    "departments_list": "https://www.iitj.ac.in/m/Index/main-departments?lg=en",
    "ai_datascience_program": "https://www.iitj.ac.in/Executive-Programs/en/B.Sc-B.S-in-Applied-AI-and-Data-Science",
    "mtech_ai_datascience": "https://www.iitj.ac.in/school-of-artificial-intelligence-data-science/en/mtech",
    "engineering_science_program": "https://www.iitj.ac.in/office-of-academics/en/engineering-science",
    "math_department": "https://www.iitj.ac.in/mathematics",
    "liberal_arts": "https://www.iitj.ac.in/school-of-liberal-arts"
}

## We have imported all the links now we will fetch data from the pages
print("Downloading pdfs...")

pdf_text_files = []

for name, url in tqdm(pdf_links.items()):

    pdf_path = f"datasets/pdf/{name}.pdf"

    try:
        r = requests.get(url, timeout=20)
        with open(pdf_path, "wb") as f:
            f.write(r.content)

        # extract_text
        text = extract_text(pdf_path)

        txt_path = f"datasets/pdf/{name}.txt"

        with open(txt_path, "w", encoding="utf-8")as f:
            f.write(text)

        pdf_text_files.append(txt_path)

    except Exception as e:
        print("PDF error: ", name, url, e)

print("PDF processing done")

print("Scrapping HTML pages")

html_text_files = []
for name, url in tqdm(html_links.items()):
    try:
        response = requests.get(url, timeout=20)

        soup = BeautifulSoup(response.text, "html.parser")

        text = soup.get_text(separator=" ", strip=True)

        txt_path = f"datasets/html/{name}.txt"

        with open(txt_path, "w", encoding = "utf-8") as f:
            f.write(text)

        html_text_files.append(txt_path)

    except Exception:
        print("HTML error: ", url)

print("Combining corpus...")

corpus = ""

all_files = pdf_text_files + html_text_files

for file in all_files:

    with open(file, "r", encoding="utf-8") as f:

        text = f.read()

        # keep only english text
        try:
            if detect(text[:1000]) == "en":
                corpus += text + "\n"
        except:
            pass


with open("iitj_raw_corpus.txt", "w", encoding="utf-8") as f:
    f.write(corpus)

print("Corpus created: iitj_raw_corpus.txt")
