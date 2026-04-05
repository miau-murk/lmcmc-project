import numpy as np
from rdkit import Chem
from .conf_calc import ConfCalc
from .angles import Conformation

mol_file = "butan.mol"

ref_conf = Chem.MolFromMolFile(mol_file, removeHs=False)

all_dih_angles = Conformation.find_unique_dihedral_angles(ref_conf) # ([((1, 2, 3, 4), angle_value)], [ ring ])
nonring_dih_angles = all_dih_angles[0] # only non-ring angles

rotatable_dih_idx = []
for dih_angle in nonring_dih_angles:
    rotatable_dih_idx.append(list(dih_angle[0]))

# print(rotatable_dih_idx)
# Usage example
calculator = ConfCalc(mol=ref_conf,
                      dir_to_xyzs="xtb_calcs/",
                      rotable_dihedral_idxs=rotatable_dih_idx)



x = 2.0
eps = 0.001

e0 = calculator.get_energy([x], req_opt=False, req_grad=True)["grads"]
e_p = calculator.get_energy([x+eps], req_opt=False, req_grad=False)['energy']
e_m = calculator.get_energy([x-eps], req_opt=False, req_grad=False)['energy']
e_pp = calculator.get_energy([x+2*eps], req_opt=False, req_grad=False)['energy']
e_mm = calculator.get_energy([x-2*eps], req_opt=False, req_grad=False)['energy']
fd_grad = (-e_pp + 8*e_p - 8*e_m + e_mm) / (12*eps)  # ~O(eps^4)

print(fd_grad)
print(e0)