import os


class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


def dump_dict_to_file(folder_path, dictionary: dict):
    with open(folder_path, "w") as w:
        for k, v in dictionary.items():
            w.write(str(k) + ": " + str(v))
            w.write("\n")

