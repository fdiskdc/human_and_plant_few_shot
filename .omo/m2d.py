
for path, header_path, mode in JOBS:
    n = process_file(path, header_path, mode)
    print('OK', path, n)
