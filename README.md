# M4 Time-series Forecasting
This repository contains an implementation of multiple approaches to solve the forecasting problem presented by M4-Competition organizers on the hourly dataset. Detailed explanation of the theoratical background can be found in the associated pdf file.

---
## Table of Contents

- [Setup](#setup)
- [Project Structure](#project-structure)
   * [Dataset](#dataset)
   * [Docs](#docs)
   * [models](#models)
   * [notebooks](#notebooks)
   * [src](#src)
- [Code Documentation](#code-documentation)

---
## Setup
------
To run this project do the following :

- Clone the repo locally :

```
  git clone https://github.com/yamen23ali/m4-lstm
```

- Activate python virtual env ( if needed ) and install requirements :

```
  conda activate $VIRTUAL_ENV_NAME
  pip install --upgrade pip
  pip install -r requirements.txt
```

---

## Project Structure

The project consists of the following folders:

### Dataset
------
This folder contains the hourly time-series data as provided by M4-Competition organizers. The **splitted** folder contains the data after it was splitted into (train, test and holdout) datasets.

### Docs
------
The **docsrc** folder contains the files requried to generate the docs. The **docs** folder contains the generated docs. In order to regenerate the docs after changes :
```
  cd docsrc
  make github
```

### models
------
This folder contains some of the models that were trained during the expirements. It consists of three folders **berken, berken_weighted, kl_divergenc** each of then contains the models resulted from training using the corresponding approach. All models that with the same number of layers are saved under the same folder. For each model we save :

- **architecture.json**: A file that describes the model architecture.
- **hyperparameters.json**: A file that contains the hyperparametrs values used during the training  of the model.
- **weight.h5**: The weights of the model after training.

Also the model folder might contain the results of points, upper bound, lower bound predictions for both test and houldout data. These results are only added if we run **predict_and_save** function.

### notebooks
------
The **notebooks** folder contains multiple notebooks that were used during the training and evaluation of the different approaches. The code in these notebook is merely function calls and initialization.

### src
------
This folder contains the whole source code used in this project.

---
## Code Documentation
------
A full documentation for the code can be found here https://yamen23ali.github.io/m4-lstm/

