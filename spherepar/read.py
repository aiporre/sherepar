import nibabel as nb

def read_nii(datapath):
    print('loading image: ', datapath)
    img = nb.load(datapath)
    data = img.get_data()
    return data

