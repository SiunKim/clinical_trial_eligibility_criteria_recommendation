import re



def split_EC_textblock(EC_textblock: str) -> list:
    EC_textblock = re.sub(r'^\n', '', EC_textblock)
    ECs_not_stripped = EC_textblock.split('\r\n\r\n')
    ECs_not_stripped = [re.sub(r'\r\n\s+', ' ', EC_ind) for EC_ind in ECs_not_stripped]
    
    #get meta-information on ECs
    hierarchies_of_ECs = find_hierarchy_of_EC_ind_in_EC_textblock(ECs_not_stripped)
    ECs_inc_exc = get_ECs_inc_exc(ECs_not_stripped)
    #finalized ECs
    ECs_concatenated, ECs_inc_exc_rev = \
        concate_ECs_by_hierarchy(ECs_not_stripped, hierarchies_of_ECs, ECs_inc_exc)
    ECs_final = finalize_ECs(ECs_concatenated, ECs_inc_exc_rev)
    
    return ECs_final


def remove_heading_EC(EC_not_stripped: str) -> list:
    EC_ind = EC_not_stripped.strip()
    EC_ind = remove_dash_from_start(EC_ind).strip()
    EC_ind = remove_digit_title(EC_ind).strip()
    
    return EC_ind


def finalize_ECs(ECs_not_stripped: list, ECs_inc_exc: list) -> list:
    ECs_final = []
    for EC_ind, EC_inc_exc in zip(ECs_not_stripped, ECs_inc_exc):
        if EC_inc_exc in ['inclusion_criteria_mark', 'exclusion_criteria_mark']:
            pass
        elif EC_ind.strip().endswith(':'):
            pass
        elif EC_ind.strip().startswith('*'):
            pass
        else:
            if not EC_ind.split()[0].isupper():
                EC_ind = EC_ind[0].lower() + EC_ind[1:]
            EC_final = ' '.join([EC_inc_exc, EC_ind]).strip()
            EC_final = re.sub(r';$', '', EC_final)
            EC_final = re.sub(r'\.$', '', EC_final)
            ECs_final.append(EC_final)
            
    return ECs_final


def get_ECs_inc_exc(ECs_not_stripped: list):
    ECs_stripped = [EC_ind.strip() for EC_ind in ECs_not_stripped]
    
    ECs_inc_exc = []; inc_exc = '[inclusion]'
    for EC_ind in ECs_stripped:
        if 'inclusion criteria' in EC_ind.lower():
            ECs_inc_exc.append('inclusion_criteria_mark')
            inc_exc = '[inclusion]'
        elif 'exclusion criteria' in EC_ind.lower():
            ECs_inc_exc.append('exclusion_criteria_mark')
            inc_exc = '[exclusion]'
        else:
            ECs_inc_exc.append(inc_exc)
            
    return ECs_inc_exc
            

def find_hierarchy_of_EC_ind_in_EC_textblock(ECs_not_stripped: list) -> list:
    ECs_space_counts = [count_space_char_from_start(EC_ind) for EC_ind in ECs_not_stripped]
    space_counts_sorted = sorted(list(set(ECs_space_counts)))    
    
    hierarchies_of_ECs = [space_counts_sorted.index(ECs_space_counts[i]) for i in range(len(ECs_not_stripped))]
    
    return hierarchies_of_ECs


def find_parent_from_hiers(hierarchies_of_ECs: list, current_idx: int):
    '''
    hierarchies_of_ECs list를 받아서 특정 index parent를 찾아 그 index를 반환 (parent가 없는 경우 'root'를 반환)
    '''
    current_hiers = hierarchies_of_ECs[current_idx]
    hiers_of_parent_hier = [idx for idx, h in enumerate(hierarchies_of_ECs[:current_idx]) 
                                if h<current_hiers]
    
    if hiers_of_parent_hier:
        parent_idx = max(hiers_of_parent_hier)
    else:
        parent_idx = 'root'
    
    return parent_idx
    

def convert_hierarchy_list_to_dict(hierarchies_of_ECs: list) -> dict:
    '''
    hierarchies_of_ECs 내 hierarchy 정보를 각기 parent-child 관계를 나타내는 dictionary로 표현
    
    Arg:
        - hierarchies_of_ECs (list)
    Output:
        - hierarchies_of_ECs_dict (dict):
         * key (int): index (0부터)
         * value (int or str-'root'): 해당 index의 parent의 index 값 (해당 index의 parent가 없는 경우는 'root)
    '''
    hierarchies_of_ECs_dict = {idx: find_parent_from_hiers(hierarchies_of_ECs, idx) 
                                for idx in range(len(hierarchies_of_ECs))}
    return hierarchies_of_ECs_dict
    
    
def concate_ECs_by_hierarchy(ECs_not_stripped: list, 
                             hierarchies_of_ECs: list, 
                             ECs_inc_exc:list) -> list:
    def concate_EC_with_parent(ECs_not_stripped, hierarchies_of_ECs_dict, EC_idx):
        current_EC = remove_heading_EC(ECs_not_stripped[EC_idx])
        
        if hierarchies_of_ECs_dict[EC_idx]!='root':
            parent_idx = hierarchies_of_ECs_dict[EC_idx]
            parent_EC = concate_EC_with_parent(ECs_not_stripped, 
                                               hierarchies_of_ECs_dict, 
                                               parent_idx)
            
            if ((parent_EC.endswith(':')) 
            and (len(parent_EC.split())>3)
            and not ('criteria' in parent_EC.lower())
            and not ('characteristics' in parent_EC.lower())
            and not ('prior concurrent' in parent_EC.lower())):
                return parent_EC + ' ' + current_EC.strip()
            else:
                return current_EC.strip()
        else:
            return current_EC.strip()
        
    hierarchies_of_ECs_dict = convert_hierarchy_list_to_dict(hierarchies_of_ECs)
    ECs_concatenated = []; ECs_inc_exc_rev = []
    for i in range(len(hierarchies_of_ECs)):
        EC_concatenated = concate_EC_with_parent(ECs_not_stripped, 
                                                 hierarchies_of_ECs_dict, 
                                                 i)
        if EC_concatenated:
            ECs_concatenated.append(EC_concatenated)
            ECs_inc_exc_rev.append(ECs_inc_exc[i])
        
    return ECs_concatenated, ECs_inc_exc_rev
            
        
def remove_dash_from_start(EC_ind: str):
    return re.sub(r'^[- ]+', '', EC_ind).lstrip()

def remove_digit_title(EC_ind: str):
    return re.sub(r'^[\d+\.\s*]+', '', EC_ind).lstrip()

def count_space_char_from_start(EC_ind: str) -> int:        
    return len(EC_ind) - len(EC_ind.lstrip())
    