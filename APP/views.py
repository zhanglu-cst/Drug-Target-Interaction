from django.shortcuts import render
from API import DTI_api
from data_process_to_fingerprint import get_data
from API import search_drug_api
from API import search_protein_api
from API.generate_image import to_image,generate_drug_images
from data_process_to_fingerprint import get_dict_ID_str


DTI_model = DTI_api.Model_API()
get_label_obj = get_data.Get_Label()
search_drug_obj = search_drug_api.search_drug()
search_protein_obj = search_protein_api.search_protein()

pad_smiles_probability_obj = get_data.Pad_smlies_probabitity()
pad_protein_probability_obj = get_data.Pad_protein_probability()

Dict_DrugID_To_Smiles = get_dict_ID_str.get_Dict_DrugID_To_Smiles()
Dict_ProteinID_To_Sequence = get_dict_ID_str.get_Dict_ProteinID_To_Sequence()



def index(request):
    return render(request,'index.html')




def DTI(request):
    M={}
    M['show_result'] = False
    if(request.POST):
        P = request.POST['P']
        D = request.POST['D']
        P = [P]
        D = [D]
        M['show_result'] = True
        try:
            value = DTI_model(P,D)
            to_image(D[0],'images_DTI','DTI')
            value = value[0]
            value = value * 100
            if(value>50):
                color = 0
            else:
                color = 1
            M['color'] = color
            value = '{:.3f}%'.format(value)
            labelM = get_label_obj(P,D)
            if(labelM['label']=='None'):
                M['label'] = 'None'
            else:
                if(labelM['label']==1):
                    M['label'] = '相互作用'
                else:
                    M['label'] = '不相互作用'
                M['set'] = labelM['set']
        except Exception as e:
            M['error'] = True
            M['error_msg'] = str(e)
            return render(request,'DTI.html',M)
        M['res'] = value

    return render(request,'DTI.html',M)

def search_drug(request):
    M = {}
    M['show_result'] = False
    if(request.POST):
        probability_threshold = request.POST['probability']
        P = request.POST['P']
        M['show_result'] = True
        try:
            float_probability = float(probability_threshold)
        except:
            M['error'] = True
            M['error_msg'] = '概率阈值请输入0-1的浮点数'
            return render(request,'search_Drug.html',M)
        if(float_probability<0 or float_probability>1):
            M['error'] = True
            M['error_msg'] = '概率阈值请输入0-1的浮点数'
            return render(request, 'search_Drug.html', M)
        smiles_probability = search_drug_obj(P,float_probability)
        lines_ans = pad_smiles_probability_obj(smiles_probability,P)
        lines_ans = generate_drug_images(lines_ans)
        M['lines_ans'] =lines_ans


    return render(request,'search_Drug.html',M)



def target_predict(request):
    M = {}
    M['show_result'] = False
    if(request.POST):
        probability_threshold = request.POST['probability']
        D = request.POST['D']
        M['show_result'] = True
        try:
            float_probability = float(probability_threshold)
        except:
            M['error'] = True
            M['error_msg'] = '概率阈值请输入0-1的浮点数'
            return render(request, 'search_Drug.html', M)
        if (float_probability < 0 or float_probability > 1):
            M['error'] = True
            M['error_msg'] = '概率阈值请输入0-1的浮点数'
            return render(request, 'search_Drug.html', M)
        try:
            protein_probability = search_protein_obj(D,float_probability)
        except Exception as e:
            M['error'] = True
            M['error_msg'] = str(e)
            return render(request, 'search_Drug.html', M)
        lines_ans = pad_protein_probability_obj(protein_probability,D)
        to_image(D,'images_search_protein','drug')
        M['lines_ans'] = lines_ans
    return render(request,'target_predict.html',M)



def show_Drug_info(request):
    M = {}
    L = []
    for i,ID in enumerate(Dict_DrugID_To_Smiles):
        drug_smiles = Dict_DrugID_To_Smiles[ID]
        line = [i,ID,drug_smiles]
        L.append(line)
    M['lines'] = L
    return render(request,'drug_base.html',M)


def show_Protein_info(request):
    M = {}
    L = []
    for i,ID in enumerate(Dict_ProteinID_To_Sequence):
        protein_sequence = Dict_ProteinID_To_Sequence[ID]
        line = [i,ID, protein_sequence]
        L.append(line)
    M['lines'] = L
    return render(request, 'protein_base.html', M)
