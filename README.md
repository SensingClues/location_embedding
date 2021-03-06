# Location embedding
Use location embeddings to enrich prediction mappings.

We represent locations by embedding tiles into a vector.
This technique is also known as loc2vec and was first described in 
[this blogpost](https://www.sentiance.com/2018/05/03/venue-mapping/).
An implementation in pytorch is described 
[here](https://medium.com/@sureshr/loc2vec-a-fast-pytorch-implementation-2b298072e1a7), with a corresponding
[GitHub repository](https://github.com/surya501/loc2vec). We provide a tensorflow impelmentation.

## Data generation
### Tile generation

We generate tiles from the [opentopomapp](https://opentopomap.org/).
Latitude and longitude coordinates for a certain region can be found using 
[this](https://www.gps-coordinates.net/) tool.
Tiles are generated by running `python src/downloader.py`.

The [zoom level](https://wiki.openstreetmap.org/wiki/Zoom_levels) depends on you use case.

## Training the network
The feature extractor consists of a pre-trained (imagenet) DenseNet121 followed by a 1x1 convolution layer,
a dense layer and a final embedding layer (with linear activations).

The model can be trained using by running `python src/main.py`.

## Visualizng the results
Results can be visualized by running `jupyter notebook notebook/tensorboard_embedding_visualization.ipynb` and
running all cells. Afterwards the projections can be visualized in Tensorboard by running 
`tensorboard --logdir logs/` in the terminal. Embeddings are found under the *PROJECTOR* tab.

*Note*: tensorboard projector documentation is written for `tensorflow==1.*`, whereas the model is trained in
`tensorflow==2.1`. As such, one will have to create a separate virtual environment for the visualization 
(see requirements_tensorboard.txt)