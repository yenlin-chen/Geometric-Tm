import torch

def norm_0to1(data):
    '''Transforms bfactor & pLDDT to a value between 0 and 1'''
    data['residue'].bfactor = torch.clip(
        - torch.log(data['residue'].bfactor + 1) / 10 + 1,
        min=0, max=1
    )
    data['residue'].pLDDT = data['residue'].pLDDT / 100
    data.ogt = data.ogt / 100
    return data

def trim_pae(data):
    del data['residue', 'pae', 'residue']
    return data

def trim_contact(data):
    del data['residue', 'contact', 'residue']
    return data

def trim_backbone(data):
    del data['residue', 'backbone', 'residue']
    return data

def trim_codir(data):
    del data['residue', 'codir', 'residue']
    return data

def trim_coord(data):
    del data['residue', 'coord', 'residue']
    return data

def trim_deform(data):
    del data['residue', 'deform', 'residue']
    return data

def trim_res1hot(data):
    del data['residue'].res1hot
    return data
