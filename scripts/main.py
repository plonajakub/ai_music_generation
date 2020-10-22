import data_services as ds

PATH_TO_DATA = '../data/bach_all_not_corrupted/bach_all_not_corrupted_data/'
PATH_TO_RESULTS = '../data/bach_all_not_corrupted/created_midis/'

data = ds.get_notes(PATH_TO_DATA + '01[ac]*.mid', True)
# print([len(part) for part in data])
# print(data[0][:10])

# ds.save_notes_to_midi(data[0], PATH_TO_RESULTS + '2fugue2_new.mid')

ds.create_dataset(data_matrix=data, batch_size=2, seq_len=100, flat=True)
