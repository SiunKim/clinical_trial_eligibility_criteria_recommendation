import os
import xml.etree.ElementTree as ET
from collections import defaultdict

from tqdm import tqdm



class ClinicalTrial():
    def __init__(self, dir_filename: str, root_dict: bool=False):
        tree = ET.parse(os.path.join(dir_filename))
        root = tree.getroot()
        root_dict = xml_to_dict(root)
        if root_dict:
            self.root_dict = root_dict
            
        #extract clinicaltrial information
        self.nct_id = root_dict['id_info']['nct_id']
        self.brief_title = root_dict['brief_title']
        self.official_title = root_dict['official_title']
        if 'brief_summary' in root_dict.keys():
            self.brief_summary = root_dict['brief_summary']['textblock']
        else:
            self.brief_summary = ''
        if 'detailed_description' in root_dict.keys():
            self.detailed_description = root_dict['detailed_description']['textblock']
        else:
            self.detailed_description = ''
        
        self.overall_status = root_dict['overall_status']
        self.start_date = root_dict['start_date']
        self.completion_date = root_dict['completion_date']
        self.overall_status = root_dict['overall_status']
        self.primary_completion_date = root_dict['primary_completion_date']
        self.has_expanded_access = root_dict['has_expanded_access']
        
        self.phase = root_dict['phase']
        self.study_type = root_dict['study_type']
        
        if 'study_design_info' in root_dict.keys():
            sdi_dict = root_dict['study_design_info']
            self.allocation = sdi_dict['allocation']
            self.intervention_model = sdi_dict['intervention_model']
            self.intervention_model_description = sdi_dict['intervention_model_description']
            self.primary_purpose = sdi_dict['primary_purpose']
            self.masking = sdi_dict['masking']
            self.masking_description = sdi_dict['masking_description']
        else:
            self.primary_outcome_measure = self.primary_outcome_time_frame = self.allocation = self.intervention_model = self.intervention_model_description = self.primary_purpose = self.masking = self.masking_description = ''
        
        if 'primary_outcome' in root_dict.keys():
            primary_outcomes = root_dict['primary_outcome'] if type(root_dict['primary_outcome'])==list else [root_dict['primary_outcome']]
            
            self.primary_outcome_measures = []
            self.primary_outcome_time_frames = []
            self.primary_outcome_time_descriptions = []
            for po in primary_outcomes:
                self.primary_outcome_measures.append(po['measure'])
                self.primary_outcome_time_frames.append(po['time_frame'])
                self.primary_outcome_time_descriptions.append(po['description'])
        else:
            self.primary_outcome_measures =  self.primary_outcome_time_frames = self.primary_outcome_time_descriptions = ['']
            
        self.number_of_arms = root_dict['number_of_arms']
        self.enrollment = root_dict['enrollment']
        self.condition = root_dict['condition']
        
        if 'intervention' in root_dict.keys():
            interventions = root_dict['intervention'] if type(root_dict['intervention'])==list else [root_dict['intervention']]
            
            self.intervention_types = []
            self.intervention_names = []
            self.intervention_descriptions = []
            for interv in interventions:
                self.intervention_types.append(interv['intervention_type'])
                self.intervention_names.append(interv['intervention_name'])
                self.intervention_descriptions.append(interv['intervention_description'])
        else:
            self.intervention_types =  self.intervention_names = self.intervention_descriptions = ['']
            
        if 'eligibility' in root_dict.keys():
            eligibility_dict = root_dict['eligibility']
            if 'criteria' in eligibility_dict.keys():
                criteria_dict = eligibility_dict['criteria']
                self.eligibility_criteria = criteria_dict['textblock']
            else:
                self.eligibility_criteria = ''
            self.eligibility_healthy_volunteers  = eligibility_dict['healthy_volunteers']
        else:
            self.eligibility_criteria = self.eligibility_healthy_volunteers = ''
        
        self.verification_date = root_dict['verification_date']
        self.study_first_submitted = root_dict['study_first_submitted']
        self.study_first_posted = root_dict['study_first_posted']
        self.last_update_submitted = root_dict['last_update_submitted']
        self.last_update_posted = root_dict['last_update_posted']
        if 'condition_browse' in root_dict.keys():
            self.condition_browse = root_dict['condition_browse']['mesh_term']
        else:
            self.condition_browse = ''
            
            
def xml_to_dict(xml_element):
    if not xml_element:
        return xml_element.text

    dict_element = defaultdict(str)
    for child in xml_element:
        child_dict = xml_to_dict(child)

        if child.tag in dict_element:
            if type(dict_element[child.tag]) != list:
                dict_element[child.tag] = [dict_element[child.tag]]
            dict_element[child.tag].append(child_dict)
        else:
            dict_element[child.tag] = child_dict

    return dict_element
    
    
# main
xml_dir = 'C:/CReSE/datasets/CTgov_raw/AllPublicXML_230315/'
CTs = []; xml_paths = []
for subdir, _, files in tqdm(os.walk(xml_dir)):
    for file in files:
        if file.endswith('.xml'):
            xml_path = os.path.join(subdir, file)
            clinicaltrial = ClinicalTrial(xml_path, root_dict=False)
            CTs.append(clinicaltrial)
            xml_paths.append(xml_path)
            
import pickle
with open('C:/CReSE/datasets/CTgov_preprocessed/AllPublicXML_230315_parsed.p', 'wb') as f:
    pickle.dump(CTs, f)
