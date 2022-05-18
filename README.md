# torchtree-tensorflow
 [![Python package](https://github.com/4ment/torchtree-tensorflow/actions/workflows/python-package.yml/badge.svg)](https://github.com/4ment/torchtree-tensorflow/actions/workflows/python-package.yml)
 
torchtree-flow is a package providing extra functionalities from [tensorflow] for [torchtree]

## Dependencies
 - [torchtree]
 - [tensorflow]

## Installation

### Get the source code
```bash
git clone https://github.com/4ment/torchtree-tensorflow
cd torchtree-tensorflow
```

### Install using pip
```bash
pip install .
```

## Features
### Discrete Gamma site model
The easiest way to use this model is to generate a json configuration file with a Weibull site model with the appropriate number of rate categories and then replace `WeibullSiteModel` with `torchflow.GammaSiteModel`:

```bash
torchtree-cli advi -i data.fa -t data.tree -C 4 > data.json
sed -i 's/WeibullSiteModel/torchflow.GammaSiteModel/' data.json
torchtree data.json
```

or in one line:
```bash
torchtree-cli advi -i data.fa -t data.tree -C 4 | sed 's/WeibullSiteModel/torchflow.GammaSiteModel/' | torchtree -
```

[torchtree]: https://github.com/4ment/torchtree
[tensorflow]: https://github.com/tensorflow/tensorflow