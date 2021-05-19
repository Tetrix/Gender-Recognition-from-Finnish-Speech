# Gender-Recognition-from-Finnish-Speech
A Transformer-based gender recognition model from Finnish speech

To install the dependencies, run `pip install -r requirements.txt`.

The model uses log filterbanks as input features. You can extract them from a `.flac` file using:

`python extract_features.py --flac_path path_to_file.flac --destination_path features.npy`

The features are saved as a Numpy array, so use the `.npy` extension when you save the features.

To do a gender classification, run:

`python evaluate_sample.py --input features.npy`
