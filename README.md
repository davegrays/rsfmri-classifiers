# rsfmri-classifiers

### To get the ABIDE-I data
- Use the download script from this [repo](https://github.com/preprocessed-connectomes-project/abide) to get the fmri data. Note this repo is written in python2.7.
- And retrieve the phenotypic csv file [here](http://fcon_1000.projects.nitrc.org/indi/abide/abide_I.html) (you need to register)
- For more info about the ABIDE dataset, see [here](http://preprocessed-connectomes-project.org/abide/index.html).

### Build your model
- Set the paths to the downloaded files in `load_abide_data.py` and run to generate the filtered and processed data
- Train and validate your model in the notebook
