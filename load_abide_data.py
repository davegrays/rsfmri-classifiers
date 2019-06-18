import pandas as pd
from pathlib import Path
import numpy as np
import logging
from proc import r2z

datadir = '/Users/dgrayson/PycharmProjects/rs-tcnn/data/abide'
in_csv = 'Phenotypic_V1_0b_preprocessed1.csv'
out_csv = 'postprocessed_metadata.csv'
out_corrmat = 'postprocessed_corrmats.npy'
out_data = 'postprocessed_timeseries.npy'
pipeline = 'ccs'
filt = 'filt_noglobal'
parc = 'rois_cc200'
failfilters = ['qc_anat_rater_2', 'qc_func_rater_2', 'qc_anat_rater_3', 'qc_func_rater_3']


def filter_load(df, min_sequence_length=100, max_sequence_length=295):
    """ This function assumes your input dataframe points to a bunch of .dat files 
    in the FILE_ID column, and that those .dat files contain a numpy-loadable 2D array
    shaped as (Time, Regions). There are other ad-hoc filters defined for motion, etc."""
    
    # filter out the rater fails
    for ff in failfilters:
        df = df[df[ff] != 'fail']

    # filter out the mean(FD) < 0.2 subjects or perc_fd < 50
    df = df[(df['func_mean_fd'] < 0.2) & (df['func_perc_fd'] < 50)]

    # only include the intersection with the files that are there
    datapath = Path(datadir) / pipeline / filt / parc
    idset = set()
    for file in datapath.iterdir():
        if file.is_file():
            idset.add(file.stem)

    # get the filtered df and corresponding traces
    df_out = pd.DataFrame(columns=df.columns)
    traces_lists = []
    z_distros_list = []
    sequence_length_counts = dict()
    regions_counts = dict()
    for idx, row in df.iterrows():
        if idx % 10 == 0:
            logging.info("row {} of {}".format(idx, df.shape[0]))

        if row['FILE_ID'] + "_" + parc in idset:
            file = str(datapath / row['FILE_ID']) + "_{}.1D".format(parc)
            traces = np.loadtxt(file)  # this is T x R
            T, R = traces.shape
            # normalize each trace by mean and stdev
            traces = (traces - traces.mean(axis=0)) / traces.std(axis=0)
            sequence_length_counts[T] = sequence_length_counts.get(T, 0) + 1
            regions_counts[R] = regions_counts.get(R, 0) + 1

            if T > min_sequence_length and not np.any(np.isnan(traces)):
                # truncate
                traces = traces[:max_sequence_length, :]
                
                # extract fisher transformed corrmat BEFORE padding
                rmat = np.dot(traces.T, traces) / T
                zmat = r2z(rmat)
                z_distro = zmat[np.nonzero(np.triu(zmat, k=1))]
                z_distros_list.append(z_distro.tolist())
                
                # now pad
                pad_width = max_sequence_length - traces.shape[0]
                traces = np.pad(traces, ((pad_width, 0), (0, 0)), 'constant')
                
                # add as 2-d list to list of traces
                traces_lists.append(traces.tolist())
                # row['numpy_array'] = traces
                
                # add to filtered metadata
                df_out = df_out.append(row)

    logging.info('sequence_length_counts')
    logging.info(sequence_length_counts)
    logging.info('regions_counts')
    logging.info(regions_counts)

    return df_out.reset_index(), np.array(traces_lists), np.array(z_distros_list)


df = pd.read_csv(str(Path(datadir) / in_csv))
df = df.reset_index()

df_filt, traces_arr, corrmats_arr = filter_load(df)

df_filt.to_csv(str(Path(datadir) / out_csv))
np.save(str(Path(datadir) / out_data), traces_arr)
np.save(str(Path(datadir) / out_corrmat), corrmats_arr)
