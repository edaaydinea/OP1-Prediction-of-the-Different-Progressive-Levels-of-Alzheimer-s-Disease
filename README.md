# Prediction of the Different Progressive Levels of Alzheimer's Disease

## Business Understanding - Project Objective

- This is an optional model development project on a real dataset related to predicting the different progressive levels of Alzheimer's disease (AD). The students are expected to use tensorflow library for modeling process and will be asked to submit predicted labels for a test dataset by which their score will be evaulated objectively. 
- This project is included in the UpSchool - Google Developers Machine Learning - Deep Learning Program. 
- In this project, you are supposed to provide a data science model to determine the level of Alzheimer disease. The levels are the ordinal categories from lower to higher respectively: 0, 0.25, 0.50, 1.0, 2.0, 3.0 (that are the progressive levels of Alzheimer's disease)
- You are expected to use the following features:

['EDUC','NACCMOCA','MARISTAT','NACCFAM','NACCGDS','NACCNE4S','NACCAPOE', 'INDEPEND','RESIDENC','ANYMEDS','NACCAMD','DEL','HALL','DEPD','ANX','APA','DISN', 'IRR','MOT','AGIT','ELAT','NITE','APP','DROPACT','NACCAGEB','SEX']

## Data Understanding

| Form                                                | Variable Name | Short Descriptor                                                  | Variable Type         | Source |
|-----------------------------------------------------|---------------|-------------------------------------------------------------------|-----------------------|--------|
| A1 Subject Demographics                             | EDUC          | Years of education                                                | Original UDS Question | v1-3   |
| A1 Subject Demographics                             | MARISTAT      | Marital Status                                                    | Original UDS Question | v1-3   |
| A1 Subject Demographics                             | INDEPEND      | Level of Independence                                             | Original UDS Question | v1-3   |
| A1 Subject Demographics                             | RESIDENC      | Type of Residence                                                 | Original UDS Question | v1-3   |
| A1 Subject Demographics                             | NACCAGEB      | Subject's age at initial visit                                    | NACC derived variable | v1-3   |
| A1 Subject Demographics                             | SEX           | Subject's sex                                                     | Original UDS Question | v1-3   |
| A3 Subject Family History                           | NACCFAM       | Indicator of first-degree family member with cognitive impairment | NACC derived variable | v1-3   |
| A4 Subject Medications                              | ANYMEDS       | Subject taking any medications                                    | Original UDS Question | v1-3   |
| A4 Subject Medications                              | NACCAMD       | Total number of medications reported at each visit                | NACC derived variable | v1-3   |
| B5 Neuropsychiatric Inventory Questionnaire (NPI-Q) | DEL           | Delusions in the last month                                       | Original UDS Question | v1-3   |
| B5 Neuropsychiatric Inventory Questionnaire (NPI-Q) | HALL          | Hallucinations in the last month                                  | Original UDS Question | v1-3   |
| B5 Neuropsychiatric Inventory Questionnaire (NPI-Q) | DEPD          | Depression or dysphoria in the last month                         | Original UDS Question | v1-3   |
| B5 Neuropsychiatric Inventory Questionnaire (NPI-Q) | ANX           | Anxiety in the last month                                         | Original UDS Question | v1-3   |
| B5 Neuropsychiatric Inventory Questionnaire (NPI-Q) | APA           | Apathy or indifference in the last month                          | Original UDS Question | v1-3   |
| B5 Neuropsychiatric Inventory Questionnaire (NPI-Q) | DISN          | Disinhibition in the last month                                   | Original UDS Question | v1-3   |
| B5 Neuropsychiatric Inventory Questionnaire (NPI-Q) | IRR           | Irritability or lability in the last month                        | Original UDS Question | v1-3   |
| B5 Neuropsychiatric Inventory Questionnaire (NPI-Q) | MOT           | Motor disturbance in the last month                               | Original UDS Question | v1-3   |
| B5 Neuropsychiatric Inventory Questionnaire (NPI-Q) | AGIT          | Agitation or aggression in the last month                         | Original UDS Question | v1-3   |
| B5 Neuropsychiatric Inventory Questionnaire (NPI-Q) | ELAT          | Elation or euphoria in the last month                             | Original UDS Question | v1-3   |
| B5 Neuropsychiatric Inventory Questionnaire (NPI-Q) | NITE          | Nighttime behaviors in the last month                             | Original UDS Question | v1-3   |
| B5 Neuropsychiatric Inventory Questionnaire (NPI-Q) | APP           | Appetite and eating problems in the last month                    | Original UDS Question | v1-3   |
| B6 Geriatric Depression Scale (GDS)                 | NACCGDS       | Total GDS Score                                                   | NACC derived variable | v1-3   |
| B6 Geriatric Depression Scale (GDS)                 | DROPACT       | Have you dropped many of your activities and interests?           | Original UDS Question | v1-3   |
| C2 Neuropsychological Battery Scores                | NACCMOCA      | MoCA Total Score - corrected for education                        | NACC derived variable | v3     |



Target : CDRGLOB  --> Global CDR
- 0.0 = No impairment
- 0.25 = Questionable impairment
- 0.5 = Mild impairment
- 1 (2 & 3) = Moderate - Severe impairment

Your train dataset size should be 70%, validation dataset size 15% as well as the test size 15%. Your target metric will be F1 score.

For MARISTAT (Marital Status):
- 1 : Married
- 2 : Widowed
- 3 : Divorced
- 4 : Separated
- 5 : Never married (or marriage was annulled)
- 6 : Living as married /domestic partner
- 9 : Other or unknown

For INDEPEND:
- 1 : Able to live independently
- 2 : Requires some assistance with complex activities
- 3 : Requires some assistance with basic activities
- 4 : Completely dependent
- 9 : Unknown

For RESIDENC:
- 1 : Single or multifamily private residence (apartment, condo, house)
- 2 : Retirement community or independent group living
- 3 : Assisted living, adult family home, or boarding home
- 4 : Skilled nursing facility, nursing home, hospital, or hospice
- 9 : Other or unknown

For EDUC :
- 0 - 36 
  - 12 : High School
  - 16 : Bachelors degree
  - 18 : Master's Degree
  - 20 : Doctorate
- 99 : Unknown

For SEX: (Convert this ):
- 1 : Male 
- 2 = Female

For NACCAGEB:
- 18 - 120 age

For NACCFAM:
- 0 = No report of a first-degree family member with cognitive impairment
- 1 = Report of at least one first-degree family member with cognitive impairment
- 9 = Unknown
- -4 = Not available: UDS form submitted did not collect data in this way, or a skip pattern precludes response to this question

For ANYMEDS:
- 0 : No
- 1 : Yes
- -4 : Did not complete medications from (NaN olarak sayabilirsin)

For NACCAMD:
- 0 - 40 
- -4 : Did not complete medications form (NaN olarak sayabilirsin.)

For DEL /HALL / AGIT / DEPD / ANX / ELAT / APA / DISN / IRR / MOT / NITE / APP:
- 0 : No
- 1 : Yes 
- 9 : Unknown
- -4 : Not available. 

For DROPACT:
- 0 : No
- 1 : Yes 
- 9 : Did not answer 
- -4 : Not available

For NACCGDS:
- 0 - 15
- 88 : Could not be calculated
- - 4 : Not available. 

For NACCMOCA:
- 0 - 30 
- 88 = Item(s) or whole test not administered 
- 99 = Years of education missing/unknown 
- -4 = Not available:
