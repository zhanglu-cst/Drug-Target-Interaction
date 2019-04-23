from rdkit.Chem import AllChem as Chem
from rdkit.Chem import Draw
import project_path
import PIL
import os


def to_image(smile,dirname, file_name):
    m = Chem.MolFromSmiles(smile)
    img = Draw.MolToImage(m)
    path = os.path.join(project_path.PROJECT_ROOT,'static')
    path = os.path.join(path,dirname)
    path = os.path.join(path,file_name+'.png')
    img.save(path)


def generate_drug_images(lines_ans):
    for i, line in enumerate(lines_ans):
        smile = line['sp'][0]
        to_image(smile,'images_search_drug',str(i))
        line['image_name'] = str(i)
    return lines_ans

