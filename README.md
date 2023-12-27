# Deep Learning with Tensorflow and Keras Coding Examples

Amita Kapoor

Antonio Gulli

Sujit Pal

## Folder Structure in VeryDeepCNN

```
├── main.py             - here's an example of main that is responsible for the whole pipeline.
│
│
├── base                - this folder contains the abstract classes of the project components
│   ├── base_data_loader.py   - this file contains the abstract class of the data loader.
│   ├── base_model.py   - this file contains the abstract class of the model.
│   └── base_train.py   - this file contains the abstract class of the trainer.
│
│
├── model               - this folder contains the models of your project.
│   └── simple_mnist_model.py
│
│
├── trainer             - this folder contains the trainers of your project.
│   └── simple_mnist_trainer.py
│
|
├── data_loader         - this folder contains the data loaders of your project.
│   └── simple_mnist_data_loader.py
│
│
├── configs             - this folder contains the experiment and model configs of your project.
│   └── simple_mnist_config.json
│
│
├── datasets            - this folder might contain the datasets of your project.
│
│
└── utils               - this folder contains any utils you need.
     ├── config.py      - util functions for parsing the config files.
     ├── dirs.py        - util functions for creating directories.
     └── utils.py       - util functions for parsing arguments.
```

# Contributing
Any contributions are welcome including improving the template and example projects.

# Acknowledgements
This project template is based on [ashishodu2023](https://github.com/ashishodu2023)'s [Tensorflow Projects](https://github.com/ashishodu2023/tensorflow).
