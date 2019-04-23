import json
import project_path
import os


dirpath = os.path.join(project_path.PROJECT_ROOT,'data_process_to_fingerprint')

Dict_DrugID_To_Smiles_path = os.path.join(dirpath,'Dict_DrugID_To_Smiles.json')
Dict_ProteinID_To_Sequence_path = os.path.join(dirpath,'Dict_ProteinID_To_Sequence.json')


def get_Dict_DrugID_To_Smiles():
    with open(Dict_DrugID_To_Smiles_path,'r') as f:
        D = json.load(f)
    return D

def get_Dict_ProteinID_To_Sequence():
    with open(Dict_ProteinID_To_Sequence_path,'r') as f:
        D = json.load(f)
    return D

