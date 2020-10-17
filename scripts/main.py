import data_services as ds

PATH_TO_DATA = '../data/bach_all_not_corrupted/bach_all_not_corrupted_data/'
PATH_TO_RESULTS = '../data/bach_all_not_corrupted/created_midis/'

data = ds.get_notes(PATH_TO_DATA + '01allema.mid')
print([len(part) for part in data])
print(data[0][:10])

ds.save_notes_to_midi(data[0], PATH_TO_RESULTS + '01allema_new.mid')

