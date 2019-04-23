import numpy as np
from rdkit import DataStructs
from rdkit.Chem import AllChem
from rdkit.Chem import AllChem as Chem


class FP:
    """
    Molecular fingerprint class, useful to pack features in pandas df

    Parameters
    ----------
    fp : np.array
        Features stored in numpy array
    names : list, np.array
        Names of the features
    """

    def __init__(self, fp, names):
        self.fp = fp
        self.names = names

    def __str__(self):
        return "%d bit FP" % len(self.fp)

    def __len__(self):
        return len(self.fp)


def get_cfps(mol, radius = 2, nBits = 2 * 1024, useFeatures = False, counts = False, dtype = np.float32):
    """Calculates circural (Morgan) fingerprint.
    http://rdkit.org/docs/GettingStartedInPython.html#morgan-fingerprints-circular-fingerprints

    Parameters
    ----------
    mol : rdkit.Chem.rdchem.Mol
    radius : float
        Fingerprint radius, default 2
    nBits : int
        Length of hashed fingerprint (without descriptors), default 1024
    useFeatures : bool
        To get feature fingerprints (FCFP) instead of normal ones (ECFP), defaults to False
    counts : bool
        If set to true it returns for each bit number of appearances of each substructure (counts). Defaults to false (fingerprint is binary)
    dtype : np.dtype
        Numpy data type for the array. Defaults to np.float32 because it is the default dtype for scikit-learn

    Returns
    -------
    ML.FP
        Fingerprint (feature) object
    """
    arr = np.zeros((1,), dtype)

    if counts is True:
        info = {}
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits = nBits, useFeatures = useFeatures,
                                                   bitInfo = info)
        DataStructs.ConvertToNumpyArray(fp, arr)
        arr = np.array([len(info[x]) if x in info else 0 for x in range(nBits)], dtype)
    else:
        DataStructs.ConvertToNumpyArray(
                AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits = nBits, useFeatures = useFeatures), arr)
    return FP(arr, range(nBits))


def get_fingerprint_from_smiles(smiles):
    m = Chem.MolFromSmiles(smiles)
    Finger = get_cfps(m)
    fp = Finger.fp
    fp = fp.tolist()
    return fp
