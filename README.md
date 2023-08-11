# latent-hierarchical
Latent, hierarchical linear gaussian models.

# Setup

Use poetry to install dependencies and run the script:
```python
poetry shell
poetry install
python main.py
```

# Usage

Specify parameters via `params.json` file in the root folder to run the procedure.

## Running predefined scenarios

Specify a new scenario in `scenarios.py`:
```python
def scenario5():
    g = GaussianGraph()
    g.add_variable("L1", None)
    g.add_variable("L2", "L1")
    g.add_variable("X1", "L1")
    g.add_variable("X2", "L1")
    g.add_variable("X3", "L2")
    g.add_variable("X4", "L2")
    return g
    
scenarios = {"5": scenario5}
```

Next, adjust parameters in `params.json` to run the desired scenario. Set `sample=false` to run the procedure assuming we have the true covariance matrix, and `sample=true` if we want to estimate the rank tests using generated data from the scenario. For `sample=true`, you can set the `alpha` parameter to adjust the threshold of rejecting the null hypothesis, and `n_samples` to indicate how many rows of data to generate.

```json
{
    "alpha": 0.005,
    "scenario": "5",
    "maxk": 3,
    "n_samples": 1000,
    "stage": 3,
    "sample": false,
}
```

## Running on your own data

To run the procedure on your own data, you can put the data into the root folder and indicate the path of the data in `params.json`. Note that `scenario` has to be set to null, and `sample` has to be set to true. The column names of the data file should all start with "X" (to indicate that these are measured variables).

```json
{
    "alpha": 0.05,
    "n_samples": 1000,
    "scenario": null,
    "data_path": "df.csv",
    "maxk": 3,
    "stage": 3,
    "sample": true,
}
```



