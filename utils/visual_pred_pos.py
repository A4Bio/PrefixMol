import os
from shutil import copy
from utils.data_structs_exp import format_str

#visualize the pred_pos, we can comment those lines later

def visual_pred_pos(protein_pdb,ligand_pred_pos_list):
    for i, ligand_pos_state in enumerate(ligand_pred_pos_list):
        tmp_text =[]
        idx = 1
        for ligand_pos in ligand_pos_state:
            atom_name = "  "
            res = "  "
            chain_id = "  "
            res_seq = "  "
            x, y, z = float(ligand_pos[0]), float(ligand_pos[1]), float(ligand_pos[2])
            elem = "C"
            occu = 1.00
            temp = 55.23
            tmp_text.append(format_str(idx, atom_name, res, chain_id, res_seq, x, y, z, occu, temp, elem, "HETATM"))
            idx += 1 
        tmp_text.append("END")
        os.makedirs("pdb_pred_pos_visualization", exist_ok=True)
        copy('/huyuqi/MolDesign/data/crossdocked_pocket10/'+protein_pdb[i], "/huyuqi/MolDesign/pdb_pred_pos_visualization")
        path = protein_pdb[i].split("/")
        with open('/huyuqi/MolDesign/pdb_pred_pos_visualization/'+path[1], "a", encoding='gbk') as f:
            f.writelines("\n".join(tmp_text))
        f.close()
    
    return None