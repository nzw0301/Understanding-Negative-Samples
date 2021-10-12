from collections import OrderedDict

import torch
from torchvision.models import resnet18, resnet50


class NonLinearClassifier(torch.nn.Module):
    def __init__(self, num_features: int = 128, num_hidden: int = 128, num_classes: int = 10):
        super(NonLinearClassifier, self).__init__()

        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(num_features, num_hidden),
            torch.nn.BatchNorm1d(num_hidden),
            torch.nn.ReLU(),
            torch.nn.Linear(num_hidden, num_classes),
        )

    def forward(self, inputs: torch.FloatTensor) -> torch.FloatTensor:
        """
        Return Unnormalized probabilities

        :param inputs: Mini-batches of feature representation.
        :return: Unnormalized probabilities.
        """

        return self.classifier(inputs)  # N x num_classes


class NormalisedLinear(torch.nn.Linear):
    """
    Linear module with normalized weights.
    """

    def forward(self, input) -> torch.FloatTensor:
        w = torch.nn.functional.normalize(self.weight, dim=1, p=2)
        return torch.nn.functional.linear(input, w, self.bias)


class LinearClassifier(torch.nn.Module):
    def __init__(self, num_features: int = 128, num_classes: int = 10, normalize: bool = True):
        """
        Linear classifier for linear evaluation protocol.

        :param num_features: The dimensionality of feature representation
        :param num_classes: The number of supervised class
        :param normalize: Whether feature is normalized or not.
        """

        super(LinearClassifier, self).__init__()
        if normalize:
            self.classifier = NormalisedLinear(num_features, num_classes, bias=False)
        else:
            self.classifier = torch.nn.Linear(num_features, num_classes)

    def forward(self, inputs: torch.FloatTensor) -> torch.FloatTensor:
        return self.classifier(inputs)  # N x num_classes


class CentroidClassifier(torch.nn.Module):

    def __init__(self, weights: torch.FloatTensor):
        """
        :param weights: The pre-computed weights of the classifier.
        """
        super(CentroidClassifier, self).__init__()
        self.weights = weights  # d x num_classes

    def forward(self, inputs) -> torch.FloatTensor:
        return torch.matmul(inputs, self.weights)  # N x num_classes

    @staticmethod
    def create_weights(data_loader, num_classes: int) -> torch.FloatTensor:
        """
        :param data_loader: Data loader of feature representation to create weights.
        :param num_classes: The number of classes.
        :return: FloatTensor contains weights.
        """

        X = data_loader.data
        Y = data_loader.targets

        weights = []
        for k in range(num_classes):
            ids = torch.where(Y == k)[0]
            weights.append(torch.mean(X[ids], dim=0))

        weights = torch.stack(weights, dim=1)  # d x num_classes
        return weights


class ProjectionHead(torch.nn.Module):
    def __init__(self, num_last_hidden_units: int, d: int):
        """
        :param num_last_hidden_units: the dimensionality of the encoder's output representation.
        :param d: the dimensionality of output.

        """
        super(ProjectionHead, self).__init__()

        self.projection_head = torch.nn.Sequential(OrderedDict([
            ('linear1', torch.nn.Linear(num_last_hidden_units, num_last_hidden_units)),
            ('bn1', torch.nn.BatchNorm1d(num_last_hidden_units)),
            ('relu1', torch.nn.ReLU()),
            ('linear2', torch.nn.Linear(num_last_hidden_units, d, bias=False))
        ]))

    def forward(self, inputs: torch.FloatTensor) -> torch.FloatTensor:
        return self.projection_head(inputs)


class ContrastiveModel(torch.nn.Module):

    def __init__(self, base_cnn: str = "resnet18", d: int = 128, is_cifar: bool = True):
        """
        :param base_cnn: The backbone's model name. resnet18 or resnet50.
        :param d: The dimensionality of the output feature.
        :param is_cifar:
            model is for CIFAR10/100 or not.
            If it is `True`, network is modified by following SimCLR's experiments.
        """

        assert base_cnn in {"resnet18", "resnet50"}
        super(ContrastiveModel, self).__init__()

        if base_cnn == "resnet50":
            self.f = resnet50()
            num_last_hidden_units = 2048
        elif base_cnn == "resnet18":
            self.f = resnet18()
            num_last_hidden_units = 512

            if is_cifar:
                # replace the first conv2d with smaller conv
                self.f.conv1 = torch.nn.Conv2d(
                    in_channels=3, out_channels=64, stride=1, kernel_size=3, padding=3, bias=False
                )

                # remove the first max pool
                self.f.maxpool = torch.nn.Identity()
        else:
            raise ValueError(
                "`base_cnn` must be either `resnet18` or `resnet50`. `{}` is unsupported.".format(base_cnn)
            )

        # drop the last classification layer
        self.f.fc = torch.nn.Identity()

        # non-linear projection head
        self.g = ProjectionHead(num_last_hidden_units, d)

    def encode(self, inputs: torch.FloatTensor) -> torch.FloatTensor:
        """
        return features before projection head.
        :param inputs: FloatTensor that contains images.
        :return: feature representations.
        """

        return self.f(inputs)  # N x num_last_hidden_units

    def forward(self, inputs: torch.FloatTensor) -> torch.FloatTensor:

        h = self.encode(inputs)
        z = self.g(h)
        return z  # N x d


class SupervisedModel(torch.nn.Module):

    def __init__(self, base_cnn: str = "resnet18", num_classes: int = 10, is_cifar: bool = True):
        """
        :param base_cnn: name of backbone model.
        :param num_classes: the number of supervised classes.
        :param is_cifar: Whether CIFAR10/100 or not.
        """

        assert base_cnn in {"resnet18", "resnet50"}
        super(SupervisedModel, self).__init__()

        if base_cnn == "resnet50":
            self.f = resnet50()
            num_last_hidden_units = 2048
        elif base_cnn == "resnet18":
            self.f = resnet18()
            num_last_hidden_units = 512

            if is_cifar:
                # replace the first conv2d with smaller conv
                self.f.conv1 = torch.nn.Conv2d(
                    in_channels=3, out_channels=64, stride=1, kernel_size=3, padding=3, bias=False
                )

                # remove the first max pool
                self.f.maxpool = torch.nn.Identity()
        else:
            raise ValueError(
                "`base_cnn` must be either `resnet18` or `resnet50`. `{}` is unsupported.".format(base_cnn)
            )

        self.f.fc = torch.nn.Linear(num_last_hidden_units, num_classes)

    def forward(self, inputs: torch.FloatTensor) -> torch.FloatTensor:

        return self.f(inputs)


class SupervisedFastText(torch.nn.Module):

    def __init__(self, num_embeddings: int, embedding_dim: int = 10, num_classes: int = 4) -> None:
        """
        :param num_embeddings: The size of vocabulary
        :param embedding_dim: the dimensionality of feature representations.
        :param num_classes: the number of supervised classes.
        """

        super(SupervisedFastText, self).__init__()

        self._num_embeddings = num_embeddings
        self._embedding_dim = embedding_dim

        self.f = torch.nn.Sequential(OrderedDict([
            ("embeddings",
             torch.nn.EmbeddingBag(num_embeddings=num_embeddings, embedding_dim=embedding_dim, sparse=False)),
            ("fc", torch.nn.Linear(embedding_dim, num_classes, bias=False)),
        ]))

        self.reset_parameters()

    def reset_parameters(self) -> None:
        """
        fasttext's parameter initialization.
        """
        # https://github.com/facebookresearch/fastText/blob/25d0bb04bf43d8b674fe9ae5722ef65a0856f5d6/src/fasttext.cc#L669
        upper = 1. / self._embedding_dim
        self.f.embeddings.weight.data.uniform_(-upper, upper)

        # https://github.com/facebookresearch/fastText/blob/25d0bb04bf43d8b674fe9ae5722ef65a0856f5d6/src/fasttext.cc#L677
        self.f.fc.weight.data.zero_()

    def forward(self, inputs: torch.Tensor, offsets: torch.Tensor) -> torch.Tensor:
        return self.f.fc(self.f.embeddings(inputs, offsets))


class ContrastiveFastText(torch.nn.Module):

    def __init__(self, num_embeddings: int, embedding_dim: int = 10, num_last_hidden_units: int = 128,
                 with_projection_head=True) -> None:
        """
        :param num_embeddings: The size of vocabulary
        :param embedding_dim: the dimensionality of feature representations.
        :param num_last_hidden_units: the number of units in the final layer. If `with_projection_head` is False,
            this value is ignored.
        :param with_projection_head: bool flag whether or not to use additional linear layer whose dimensionality is
            `num_last_hidden_units`.
        """

        super(ContrastiveFastText, self).__init__()

        self._num_embeddings = num_embeddings
        self._embedding_dim = embedding_dim
        self._num_last_hidden_units = num_last_hidden_units
        self._with_projection_head = with_projection_head

        self.f = torch.nn.EmbeddingBag(num_embeddings=num_embeddings, embedding_dim=embedding_dim, sparse=False)

        if self._with_projection_head:
            self.g = ProjectionHead(num_last_hidden_units, embedding_dim)
        else:
            self.g = torch.nn.Identity()

    def encode(self, inputs: torch.Tensor, offsets: torch.Tensor) -> torch.Tensor:
        """
        taken inputs and its offsets, extract feature representation.
        """
        return self.f(inputs, offsets)  # (B, embedding_dim)

    def forward(self, inputs: torch.Tensor, offsets: torch.Tensor) -> torch.Tensor:
        """
        taken inputs and its offsets, extract feature representation, then apply additional feature transform
        to calculate contrastive loss.
        """
        h = self.encode(inputs, offsets)
        z = self.g(h)
        return z  # (B, embedding_dim) or (B, num_last_hidden_units) depending on `with_projection_head`
