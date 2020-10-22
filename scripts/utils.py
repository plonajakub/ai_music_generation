def log(log_type, msg):
	if log_type == 'i':
		print('[I] ' + msg)
	elif log_type == 'd':
		print('[D] ' + msg)
	else:
		print('[-] ' + msg)


def split_input_target(chunk):
	input_seq = chunk[:-1]
	target_seq = chunk[1:]
	return input_seq, target_seq
