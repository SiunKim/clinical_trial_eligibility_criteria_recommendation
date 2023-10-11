import random; random.seed(42)

import numpy as np
import pickle
from tqdm import tqdm

from datetime import datetime
import time

from AllPublicXML_preprocessing import ClinicalTrial



def convert_date_to_datetime(date_str):
    if type(date_str)==datetime:
        date_as_dt = date_str
    else: #str
        if date_str:
            if ',' in date_str:
                date_as_dt = datetime.strptime(date_str, '%B %d, %Y')
            else:  
                date_as_dt = datetime.strptime(date_str + ' 1', '%B %Y %d')
        else:
            date_as_dt = datetime.strptime('January 1900 1', '%B %Y %d')    
        
    return date_as_dt    


def get_ct_info_str(CT_ind):
    condition = CT_ind.condition
    condition = condition.lower() if type(condition)==str else [c.lower() for c in condition]
    intervention_names = CT_ind.intervention_names
    intervention_names = [i.lower() for i in intervention_names]
    poms = CT_ind.primary_outcome_measures
    poms = [p[0].lower() + p[1:] for p in poms] if poms!=[''] else 'not available'
    phase = CT_ind.phase
    phase = phase.lower() if phase not in ['', 'N/A'] else 'not availabe'
    enrollment = CT_ind.enrollment 
    enrollment = enrollment if enrollment else 'not specified'
    ehv = CT_ind.eligibility_healthy_volunteers
    ehv = 'Yes' if ehv=='Accepts Healthy Volunteers' else 'No'
    
    ct_info_str = f"Investigated condition(s): {condition if type(condition)==str else '; '.join(condition)}, Investigational drug(s)/treatment(s): {'; '.join(intervention_names) if type(intervention_names)==list else intervention_names}, Study phase: {phase}, Patient enrollment: {enrollment}, Including healthy volunteers: {ehv}, Primary outcome measure(s): {'; '.join(poms[0:3]) if type(poms)==list else poms}"
        
    return ct_info_str


# main
#CT selection settings
CTSS = {
    'WITH_DATE_FIRST_POSTED': True,
    'WITH_BRIEF_SUMMARY': True,
    'ONLY_INTERVENTINOAL': True,
    'WITH_TITLE': True,
    'ONLY_DRUG_BIOLOGICAL': True,
    'EXCLUDE_FEW_CRIT_CT': True,
    'EXCLUDE_TOO_SHORT_LONG_EC': True,
    'FIRST_POSTED_AFTER_20020503': True
}

#import CTs_parsed - 188.5seconds
start = time.time()
print('Start reading AllPublicXML_230315_parsed pickel file!')
with open('AllPublicXML_230315_parsed.p', 'rb') as f:
    CTs_parsed = pickle.load(f)
end = time.time()
print(f'Finish reading AllPublicXML_230315_parsed pickel file! (elpased time: {(end - start):.1f} seconds)')

#exclude CTs without date information (study_first_posted)
if CTSS['WITH_DATE_FIRST_POSTED']:
    CTs_parsed = [CT_ind for CT_ind in CTs_parsed if CT_ind.study_first_posted]
    print(f'1. {len(CTs_parsed)}')
#exclude CTs without brief_summaries
if CTSS['WITH_BRIEF_SUMMARY']:
    CTs_parsed = [CT_ind for CT_ind in CTs_parsed if CT_ind.brief_summary]
    print(f'2. {len(CTs_parsed)}')
#include only interventinoal CTs
if CTSS['ONLY_INTERVENTINOAL']:
    CTs_parsed = [CT_ind for CT_ind in CTs_parsed if CT_ind.study_type=='Interventional']
    print(f'3. {len(CTs_parsed)}')
#include CTs with title (brief or official)
if CTSS['WITH_TITLE']:
    CTs_parsed = [CT_ind for CT_ind in CTs_parsed if (bool(CT_ind.brief_title) or bool(CT_ind.official_title))]
    print(f'4. {len(CTs_parsed)}') 
#include only CTs for 'Drug' or 'Biological' (intervention_types)
if CTSS['ONLY_DRUG_BIOLOGICAL']:
    CTs_parsed = [CT_ind for CT_ind in CTs_parsed if(('Drug' in CT_ind.intervention_types) or ('Biological' in CT_ind.intervention_types))]
    print(f'5. {len(CTs_parsed)}')
#first posted after datetime.datetime(2002, 5, 3, 0, 0)
if CTSS['FIRST_POSTED_AFTER_20020503']:
    CTs_parsed = [CT_ind for CT_ind in CTs_parsed 
                    if (convert_date_to_datetime(CT_ind.study_first_posted)
                        >datetime(2002, 5, 3, 0, 0))] 
    print(f'6. {len(CTs_parsed)}')

#convert date attributes as datetime type
for CT_ind in tqdm(CTs_parsed):
    CT_ind.study_first_posted = convert_date_to_datetime(CT_ind.study_first_posted)
    
signal_tokens = ['inclusion', 'criteria', 'exclusion', 'characteristics']
CTs_parsed = [CT_ind for CT_ind in CTs_parsed if any([(signal_token in ' '.join(CT_ind.eligibility_criteria.strip().lower().split()[0:2])) for signal_token in signal_tokens])]; print(f'7. {len(CTs_parsed)}')

#EC preprocessing
from EC_preprocessing import split_EC_textblock
EC_textblocks = [CT_ind.eligibility_criteria for CT_ind in CTs_parsed]

print('Start parsing EC_textblocks!')
ECs = []
for etb in tqdm(EC_textblocks):
    ECs.append(split_EC_textblock(etb))
print('Finish parsing EC_textblocks!')

#get ct_info_str from CT_ind information
for CT_ind in CTs_parsed:
    CT_ind.ct_info_str = get_ct_info_str(CT_ind)

#EC selection criteria: 
if CTSS['EXCLUDE_TOO_SHORT_LONG_EC']:
    ECs = [[EC_ind for EC_ind in ECs_ind if len(EC_ind.split())>3] for ECs_ind in ECs]
    ECs = [[EC_ind for EC_ind in ECs_ind if len(EC_ind)<353] for ECs_ind in ECs]; print(f'10. ECs len - {len(ECs)}')
    ECs_unlisted = [EC_ind for ECs_ind in ECs for EC_ind in ECs_ind]; print(f'11. ECs total numb - {len(ECs_unlisted)}')
assert len(CTs_parsed)==len(ECs), 'The length of CTs_parsed and ECs must be always same!'
assert len(CTs_parsed)==len(ECs), 'The length of CTs_parsed and ECs must be always same!'

if CTSS['EXCLUDE_FEW_CRIT_CT']:
    CTs_ECs_parsed = [(CT_ind, ECs_ind) for CT_ind, ECs_ind in zip(CTs_parsed, ECs) 
                        if len(ECs_ind)>=2]
    CTs_parsed = [cep[0] for cep in CTs_ECs_parsed]; print(f'8. CTs_parsed len - {len(CTs_parsed)}')
    ECs = [cep[1] for cep in CTs_ECs_parsed]; print(f'9. ECs len - {len(ECs)}')
assert len(CTs_parsed)==len(ECs), 'The length of CTs_parsed and ECs must be always same!'

print('Start extracting train data tuples!')
assert len(CTs_parsed)==len(ECs), 'The length of CTs_parsed and ECs must be always same!'
#extract train data
import re
train_data_tuples = [] #list of tuple: [(EC_ind, title, nct_id, brief_summary, first_posted_date)]
for CT_ind, ECs_ind in tqdm(zip(CTs_parsed, ECs)):
    title = CT_ind.official_title if bool(CT_ind.official_title) else CT_ind.brief_title
    nct_id = CT_ind.nct_id
    brief_summary = re.sub(r'\r\n[\s]{2,}', ' ', CT_ind.brief_summary).replace('\n      ', '')
    first_posted_date = CT_ind.study_first_posted
    ct_info_str = CT_ind.ct_info_str
    
    for EC_ind in ECs_ind:
        train_data_tuples.append((EC_ind, title, nct_id, brief_summary, first_posted_date, ct_info_str))
random.shuffle(train_data_tuples)

print('Finish extracting training data tuples!')
    
print('Save train data tuples before/after 2021 Sep as pickle file!')
today = datetime.today().strftime('%y%m%d')

#save whole tdt
ECs_numb = len(train_data_tuples); ECs_numb
CTs_set = set([tdt[2] for tdt in train_data_tuples]); CTs_numb = len(CTs_set); CTs_numb
with open(f'train_data_tuples_whole_EC{ECs_numb}_CT{CTs_numb}_{today}.p', 'wb') as f:
    pickle.dump(train_data_tuples, f)

#classify based on study_first_submitted
Sep_2021 = datetime.strptime("September 2021 30", '%B %Y %d')
train_data_tuples_before_2021_Sep = [tdt for tdt in train_data_tuples if tdt[4]<=Sep_2021]; len(train_data_tuples_before_2021_Sep)
train_data_tuples_after_2021_Sep = [tdt for tdt in train_data_tuples if tdt[4]>Sep_2021]; len(train_data_tuples_after_2021_Sep)

#split train_data_tuples in terms of CT
import random; random.seed(42)
#before 2021 Sep
CTs_set = set([tdt[2] for tdt in train_data_tuples_before_2021_Sep])
CTs_set_valid = set(random.sample(list(CTs_set), 5000))
CTs_set_train = CTs_set - CTs_set_valid
CTs_set_test = set(random.sample(list(CTs_set_train - CTs_set_valid), 5000))
CTs_set_train = CTs_set_train - CTs_set_test

tdtb2021Sep_train = [tdt for tdt in train_data_tuples_before_2021_Sep if tdt[2] in CTs_set_train]
tdtb2021Sep_valid = [tdt for tdt in train_data_tuples_before_2021_Sep if tdt[2] in CTs_set_valid]
tdtb2021Sep_test = [tdt for tdt in train_data_tuples_before_2021_Sep if tdt[2] in CTs_set_test]

ECs_numb = len(tdtb2021Sep_train); ECs_numb
CTs_set = set([tdt[2] for tdt in tdtb2021Sep_train]); CTs_numb = len(CTs_set); CTs_numb
with open(f'train_data_tuples_before_2021_Sep_train_EC{ECs_numb}_CT{CTs_numb}_{today}.p', 'wb') as f:
    pickle.dump(tdtb2021Sep_train, f)
ECs_numb = len(tdtb2021Sep_valid); ECs_numb
CTs_set = set([tdt[2] for tdt in tdtb2021Sep_valid]); CTs_numb = len(CTs_set); CTs_numb
with open(f'train_data_tuples_before_2021_Sep_valid_EC{ECs_numb}_CT{CTs_numb}_{today}.p', 'wb') as f:
    pickle.dump(tdtb2021Sep_valid, f)
ECs_numb = len(tdtb2021Sep_test); ECs_numb
CTs_set = set([tdt[2] for tdt in tdtb2021Sep_test]); CTs_numb = len(CTs_set); CTs_numb
with open(f'train_data_tuples_before_2021_Sep_test_EC{ECs_numb}_CT{CTs_numb}_{today}.p', 'wb') as f:
    pickle.dump(tdtb2021Sep_test, f)
with open(f'train_data_tuples_before_2021_Sep_{today}_CT_selection_setting.txt', 'w') as f:
    f.write(str(CTSS))
    
#after 2021 Sep
CTs_set = set([tdt[2] for tdt in train_data_tuples_after_2021_Sep]); CTs_numb = len(CTs_set); CTs_numb
CTs_set_test = set(random.sample(list(CTs_set), 5000))
CTs_set_train = CTs_set - CTs_set_test

tdta2021Sep_train = [tdt for tdt in train_data_tuples_after_2021_Sep if tdt[2] in CTs_set_train]
tdta2021Sep_test = [tdt for tdt in train_data_tuples_after_2021_Sep if tdt[2] in CTs_set_test]

#after 2021 Sep
ECs_numb = len(tdta2021Sep_train); ECs_numb
CTs_set = set([tdt[2] for tdt in tdta2021Sep_train]); CTs_numb = len(CTs_set); CTs_numb
with open(f'train_data_tuples_after_2021_Sep_train_EC{ECs_numb}_CT{CTs_numb}_{today}.p', 'wb') as f:
    pickle.dump(tdta2021Sep_train, f)
    
ECs_numb = len(tdta2021Sep_test); ECs_numb
CTs_set = set([tdt[2] for tdt in tdta2021Sep_test]); CTs_numb = len(CTs_set); CTs_numb
with open(f'train_data_tuples_after_2021_Sep_test_EC{ECs_numb}_CT{CTs_numb}_{today}.p', 'wb') as f:
    pickle.dump(tdta2021Sep_test, f)

#total train dataset
tdt_train_total = tdtb2021Sep_train + tdta2021Sep_train; random.shuffle(tdt_train_total)
ECs_numb = len(tdt_train_total); ECs_numb
CTs_set = set([tdt[2] for tdt in tdt_train_total]); CTs_numb = len(CTs_set); CTs_numb
with open(f'train_data_tuples_train_total_EC{ECs_numb}_CT{CTs_numb}_{today}.p', 'wb') as f:
    pickle.dump(tdt_train_total, f)
    
print('End of all the procesess for train data preperation!')