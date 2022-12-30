# rsfmri-classifiers

### Autism classifiers with fMRI data
- This repo is intended as a demo on training / cross-validating models in Keras
- Uses publicly available MR scans from young children (2-5 years)
- Target is Autism diagnosis (binary class label)
- Features are timeseries of 200 regions
- Small sample size (<1000 scans), data quality is suspect...
- Explores linear, MLP, LSTM, and CNN models

### To get the ABIDE-I data
- The data is public but requires registration
- Use the download script from this [repo](https://github.com/preprocessed-connectomes-project/abide) to get the fmri data. Note this repo is written in python2.7.
- And retrieve the phenotypic csv file [here](http://fcon_1000.projects.nitrc.org/indi/abide/abide_I.html) (you need to register)
- For more info about the ABIDE dataset, see [here](http://preprocessed-connectomes-project.org/abide/index.html).
- Set the paths to the downloaded files in `load_abide_data.py` and run to generate the filtered and processed data

### Build your model
- Train and validate your model in the notebook
