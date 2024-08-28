import torch


class config:
    """
    A configuration class that contains various hyperparameters and settings
    for the model and training process.

    Attributes:
    -----------
    device : torch.device
        The device on which computations will be performed (either CUDA if
        available or CPU).

    factor : int
        A scaling factor used in the model. This can be adjusted based on the
        initial regularization method.

    initial_regularization_method : str
        The method of regularization to be applied initially. Options include:
        - "tv" for Total Variation
        - "ell1" for L1-norm regularization (commented out by default).

    learning_rate : float
        The learning rate for the optimizer.

    len_train : int
        The number of training samples.

    len_test : int
        The number of test samples.

    epochs : int
        The number of epochs for training the model.
    """

    device: torch.device = torch.device(
        'cuda' if torch.cuda.is_available() else 'cpu'
    )
    factor: int = 10
    initial_regularization_method: str = "tv"
    # factor = 10000
    # initial_regularization_method = "ell1"

    learning_rate: float = 1e-4
    len_train: int = 500
    len_test: int = 100
    epochs: int = 50
