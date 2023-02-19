import pandas as pd
import numpy as np
import quaternion
import os, fnmatch

# Parse data
COMMENT = '/'
DELIMITER = '\t'
QUATCOLS = ["Quat_q0", "Quat_q1", "Quat_q2", "Quat_q3"]

SPLIT_KEY = '-'
THORAX_ID = "000_00B4A9E6"
RUPPERARM_ID = "000_00B4AF58"
LUPPERARM_ID = "000_00B4A7C9"
THRESHOLD = 60
SAMPLETIME = 0.01



def read_text_file(file_path):
    df = pd.read_table(file_path, delimiter=DELIMITER,
                       comment=COMMENT, usecols=QUATCOLS)
    return df


def cal_elevation_angle(df_ref: pd.DataFrame, df_seg: pd.DataFrame) -> np.ndarray:
    ref = df_ref[QUATCOLS].to_numpy()
    seg = df_seg[QUATCOLS].to_numpy()
    ref = quaternion.as_quat_array(ref)
    seg = quaternion.as_quat_array(seg)
    res = np.multiply(np.conjugate(seg), ref)
    mat = quaternion.as_rotation_matrix(res)
    sol = np.rad2deg(np.arccos(mat[:, 2, 2]))
    return sol


def cal_elevate_stat(angle_array: np.ndarray):
    te = (angle_array > THRESHOLD).sum() * SAMPLETIME
    tt = angle_array.size * SAMPLETIME
    pt = (te/tt) * 100
    return (te, tt, pt)



rootdir = 'path/to/dir'
result = []
for rootdir, dirs, files in os.walk(rootdir):
    for subdir in dirs:
        path = os.path.join(rootdir, subdir)
        for subrootdir, subdirs, subfiles in os.walk(path):
            for name in subfiles:
                df_ref = None
                df_rseg = None
                df_lseg = None
                if name.endswith(".txt"):
                    if fnmatch.fnmatch(name, THORAX_ID):
                        df_ref = read_text_file(os.path.join(subrootdir, name))
                    if fnmatch.fnmatch(name, RUPPERARM_ID):
                        df_rseg = read_text_file(os.path.join(subrootdir, name))
                    if fnmatch.fnmatch(name, LUPPERARM_ID):
                        df_lseg = read_text_file(os.path.join(subrootdir, name))
                if None not in (df_ref , df_rseg, df_lseg):
                    right_angle = cal_elevation_angle(df_ref, df_rseg)
                    right_stat = cal_elevate_stat(right_angle)
                    left_angle = cal_elevation_angle(df_ref, df_lseg)
                    left_stat = cal_elevate_stat(left_angle)
        result.append([subdir, right_stat, left_stat])