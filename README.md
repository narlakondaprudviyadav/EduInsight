# EduInsight — Student Performance Decision Dashboard

## 1. Project Overview
EduInsight converts student-performance data into a decision-oriented analytics dashboard. It follows the Business Intelligence flow discussed in the masterclass: **data → information → insights → decisions → action**.

The dashboard provides:
- Executive KPIs
- Score distribution and trends
- Study-hours and attendance analysis
- Academic-driver analysis
- Student support-risk analysis
- Random Forest prediction model
- Individual score prediction
- Recommended actions for decision makers

## 2. Problem Statement
Educational institutions often have student data but need a simple way to convert that data into actionable insights. EduInsight helps identify performance patterns, potential support groups, important drivers, and practical intervention actions.

## 3. Dataset
Dataset: **Student Performance Factors**

Source:
https://www.kaggle.com/datasets/lainguyn123/student-performance-factors

The dataset contains student academic, behavioral, family, school and lifestyle variables, with `Exam_Score` as the target.

**Important:** The masterclass specifically says not to use the exact learning dataset used during the internship. Confirm that this dataset is different from your cohort's learning dataset before submission.

Download `StudentPerformanceFactors.csv` from the source and place it next to `eduinsight.py`.

## 4. Technology Stack
- Python
- Streamlit
- Pandas
- NumPy
- Plotly
- Scikit-learn
- Random Forest Regression

## 5. Files
- `eduinsight.py` — single combined application code
- `requirements.txt` — Python dependencies
- `README.md` — project overview and setup
- `Project_Report.docx` — documentation/report

## 6. How to Run
Open a terminal in the project folder:

```bash
pip install -r requirements.txt
streamlit run eduinsight.py
```

The browser should open the dashboard automatically. If not, open the local URL shown in the terminal.

## 7. Business Intelligence Logic
### KPIs
- Number of students
- Average exam score
- Average attendance
- Number of high-support students

### Trends / Analysis
- Exam score distribution
- Average score by motivation
- Study hours vs exam score
- Average score by resource access

### Drivers
The Random Forest model estimates which variables are most useful for predicting exam score.

### Risks
Students below a configurable score threshold are surfaced for support.

### Opportunities / Actions
The dashboard suggests practical actions around attendance, study planning, previous performance, resource access and motivation.

## 8. Submission Checklist
The session explained that the final submission consists of:
1. One code file (`.py` or `.ipynb`)
2. One `requirements.txt`
3. One project report document/PDF
4. One `README.md`
5. GitHub repository link

No ZIP is required by the stated submission process.

## 9. GitHub Quick Upload
```bash
git init
git add .
git commit -m "Add EduInsight student performance BI project"
git branch -M main
git remote add origin YOUR_GITHUB_REPOSITORY_URL
git push -u origin main
```

## 10. Responsible Use
This is a decision-support prototype. Model predictions should not be treated as definitive judgments about students. Use them with human review and institutional context.
