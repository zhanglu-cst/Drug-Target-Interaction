


amino_acids = ['A','R','D','C','Q','E','H','I','G','N','L','K','M','F','P','S','T','W','Y','V']

M = {}
index = 0

for ch in amino_acids:
    M[ch]=index
    index+=1

for i in range(20):
    for j in range(20):
        ch1 = amino_acids[i]
        ch2 = amino_acids[j]
        ch = ch1+ch2
        M[ch]=index
        index+=1

for i in range(20):
    for j in range(20):
        for k in range(20):
            ch1 = amino_acids[i]
            ch2 = amino_acids[j]
            ch3 = amino_acids[k]
            ch = ch1+ch2+ch3
            M[ch]=index
            index+=1

# print(M)

def get_fingerprint_from_protein_squeeze(seq):
    # return:type list,fingerprint
    res = [0] * 8420

    for length in range(1,4):
        for i in range(len(seq)-length):
            ch = seq[i:i+length]
            try:
                id = M[ch]
                res[id]+=1
            except:
                pass
    return res




