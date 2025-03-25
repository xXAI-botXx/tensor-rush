# Tensor RUSH

C++ Deep Learning Framework


<img src="./res/cover.png"></img>



Table of Contents:
- [Installation](#installation)
- [Usage](#usage)
- [Documentation](#documentation)
- [Planning](#planning)
- [Ressource](#ressource)






---
### Installation

Coming soon...






---
### Usage

Coming soon...






---
### Documentation

The documentation is created with Doxygen,[see here for more details](https://github.com/xXAI-botXx/Project-Helper/blob/main/guides/Doxygen.md). You find the documentation here ADD_THE_LINK_HERE.






---
### Planning



#### Usage 

**Loading Data**
```cpp
#pragma once

#include "tensor-rush.h"

class MyDataClass : public rush::data::Dataset { 
public:
    MyDataClass();
    size_t len() const override;
    rush::Tensor get_item(int idx) const override;
};
```

```cpp
#include "tensor-rush.h"

MyDataClass::MyDataClass() { }

size_t MyDataClass::len() const {
    // Implementation
}

rush::Tensor MyDataClass::get_item(int idx) const {
    // Implementation
}
```

**Create AI Model**
```cpp
#include "tensor-rush.h"

rush::nn::Model model;

model.add(rush::nn::Layer(128, rush::Activation::ReLU));  // ✅ Example of adding layers
model.add(rush::nn::Layer(64, rush::Activation::ReLU));
model.add(rush::nn::Layer(10, rush::Activation::Softmax));  // ✅ Output layer
```

```cpp
// Model initialization
model.initialize();
```

**Train AI Model**
```cpp
rush::data::Dataset dataset = MyDataClass();  // ✅ Load dataset

model.train(dataset, 10 /*epochs*/, 0.01 /*learning rate*/);
```

```cpp
// Training loop
for (int epoch = 0; epoch < 10; ++epoch) {
    float loss = model.train_step();
    std::cout << "Epoch " << epoch << " Loss: " << loss << std::endl;
}
```

**Evaluate AI Model**
```cpp
float accuracy = model.evaluate(dataset);
std::cout << "Accuracy: " << accuracy << "%" << std::endl;
```

```cpp
// Generate predictions
rush::Tensor input = dataset.get_item(0);
rush::Tensor output = model.predict(input);
std::cout << "Predicted values: " << output << std::endl;
```


#### Needed Functionality

-> Core Modules (Ordered by Namespace)**

| **Namespace** | **Feature** | **Description** |
|--------------|------------|----------------|
| `rush::nn`   | `Layer` | Defines individual layers (fully connected, convolutional, etc.). |
|              | `Model` | Manages layers, forward pass, and training. |
| `rush::math` | `Vector` | Implements vector operations. |
|              | `Matrix` | Implements matrix operations. |
|              | `Tensor` | Generalized n-dimensional data structure. |
| `rush::data` | `Dataset` | Abstract base class for datasets (overridden by user). |
|              | `CSVLoader` | Loads data from CSV files. |
| `rush::device` | `CPU` | CPU computation backend. |
|               | `Vulkan` | GPU computation backend. |
| `rush::optim` | `SGD` | Stochastic Gradient Descent optimizer. |
|              | `Adam` | Adam optimizer. |
| `rush::loss` | `MSELoss` | Mean Squared Error loss function. |
|              | `CrossEntropyLoss` | Cross-Entropy loss function. |






---
### Ressource

Following ressource got used during the creation process:
- Deep Learning: Das umfassende Handbuch -> [Thalia](https://www.thalia.de/shop/home/artikeldetails/A1053281203), [Amazon](https://www.amazon.de/Deep-Learning-umfassende-Handbuch-Forschungsans%C3%A4tze/dp/3958457002/ref=sr_1_1), [Heise Shop](https://shop.heise.de/deep-learning-das-umfassende-handbuch)
- Neuronale Netze programmieren mit Python -> [Amazon]https://www.amazon.de/Neuronale-Netze-programmieren-Python-Python-Crashkurs/dp/3367102547/ref=sr_1_1), [Rheinwerk Verlag](https://www.rheinwerk-verlag.de/neuronale-netze-programmieren-mit-python/)
- Vulkan Compute: High-Performance Compute Programming with Vulkan and Compute Shaders -> [Amazon](https://www.amazon.de/Vulkan-Compute-High-Performance-Programming-Shaders/dp/B0DLTCL3W8)







