"""
General utils
"""
import os

from local_settings import DATA_FOLDER

def file_row_downsample(in_file, out_file, ratio=10):
    out = open(out_file,'w')
    for idx, line in enumerate(open(in_file)):
        if not idx % ratio:
            out.write(line)
    out.close()

if __name__ == "__main__":
    in_file = os.path.join(DATA_FOLDER, "storks2012", "storks_2012_id_date_acc_behav.csv")
    out_file = os.path.join(DATA_FOLDER, "storks2012", "storks_2012_id_date_acc_behav_downsampledX30.csv")
    file_row_downsample(in_file, out_file, ratio=30)
