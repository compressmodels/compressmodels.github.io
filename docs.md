
## Product Documentation

### Purpose of `moco`
`moco` takes machine learning models (or for that matter any "learned" function) and optimizes them for computational efficiency. moco consists of a suite of algorithms that optimize models for computational efficiency, with no accuracy loss. 

moco analyzes data to find early-exit strategies to exploit.
- moco analyzes the input data (an example of an early-exit rule for a fraud detection model when the features are interpretable might be "it's fraud if the user has logged in 4 times in the last 10 minutes".)
- intermediate representations (often referred to as embeddings, activations, etc.) 

moco analyze the internal parameters (weights and biases) to find equivalent loss-less representations to represent  models
- e.g pruning.

moco analyzes the interaction of the parameters with the data/activations to find equivalent loss-less representations to represent models. 


### Form Factor
`moco` is a `pip`-installable Python package that offers a:
- command-line interface
- software library with proprietary algorithms accessible via API.
- client-side helper library that aggregates outputs from API into machine-runnable way.

### API Methods


#### Pruning
- `prune_neurons(model: MLPClassifier, data: np.ndarray) -> MLPClassifier`
- `condense_neurons(model: MLPClassifier, data: np.ndarray) -> Tuple[MLPClassifier, LogisticRegression]`

Notes:
1. `model` is uploaded as a `.joblib` file. 
2. model.activation must be "relu"
3. `condense_neurons` returns two models, the sum of which's output matches the original model.
4. `prune_neurons` returns 1 model, the output of which matches the output of the original model.

#### Early-Exiting
```python
class RoutedModel:
    base: torch.nn.Module
    conditional: torch.nn.Module
    router: torch.nn.Module
    yes_path: torch.nn.Module
    no_path: torch.nn.Module
```

- `compute_rule(model: torch.nn.Sequential, layer_name: str, data: torch.utils.data.Dataset) -> RoutedModel`

Notes: 
1. In many cases, `yes_path` is a simple constant function. 


##### Utility
- `get_layers(model: torch.nn.Sequential) -> List[str]`
- `profile_layers(model: torch.nn.Sequential, data: torch.utils.data.Dataset) -> List[Tuple[str, float]]`


##### Compatibility
- For any algorithm that returns models: the user will be able to specify the format of the output model. 
- This includes ONNX, torchscript, torch, coreml, etc.

I