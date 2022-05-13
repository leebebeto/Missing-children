import pickle

new_dict = {}

file_name = '/Users/ijeongsu/Desktop/761-testset/lag_identification_filtered.pickle'
with open(file_name, 'rb') as f:
    file = pickle.load(f)

# print(len(file.keys()))
# import pdb; pdb.set_trace()
# for k,v in file.items():
#     key = '/'.join(k.split('/')[4:])
#     value = []
#     for image in v:
#         value.append('/'.join(image.split('/')[-3:]))
#     new_dict[key] = value
#
#
# with open('/Users/ijeongsu/Desktop/761-testset/lag_identification_filtered.pickle', 'wb') as f:
#     pickle.dump(new_dict, f)