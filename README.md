# torchtree-tensorflow
 [![Testing](https://github.com/4ment/torchtree-tensorflow/actions/workflows/python-package.yml/badge.svg)](https://github.com/4ment/torchtree-tensorflow/actions/workflows/python-package.yml)

`torchtree-tensorflow` is a package that enhances the functionalities of [torchtree] by incorporating additional features from [tensorflow].

## Dependencies
 - [torchtree]
 - [tensorflow]

## Installation

### Installing from source
```bash
git clone https://github.com/4ment/torchtree-scipy
pip install torchtree-scipy/
```

## Features
### Discrete gamma site model
This model implements the discretized gamma distribution to model rate heterogeity accross sites. The gradient of this model with respect to the shape parameter is calculated with automatic differentiation.
In order to use this model the type of the site model must be changed to `torchtree_scipy.GammaSiteModel` in the JSON configuration file.

## License

Distributed under the GPLv3 License. See [LICENSE](LICENSE) for more information.

## Acknowledgements

torchtree-scipy makes use of the following libraries and tools, which are under their own respective licenses:

 - [PyTorch]
 - [tensorflow]
 - [torchtree]

[PyTorch]: https://pytorch.org
[torchtree]: https://github.com/4ment/torchtree
[tensorflow]: https://github.com/tensorflow/tensorflow