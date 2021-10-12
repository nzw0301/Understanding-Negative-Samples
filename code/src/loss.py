import torch


class NT_Xent(torch.nn.Module):
    """
    Normalised Temperature-scaled cross-entropy loss.
    """

    def __init__(
            self, temperature: float = 0.1, reduction: str = "mean", device: torch.device = torch.device("cpu")
    ):
        """
        :param temperature: Temperature parameter. The value must be positive.
        :param reduction: Same to PyTorch's `reduction` in losses.
        :param device: PyTorch's device instance.
        """

        reduction = reduction.lower()

        if temperature <= 0.:
            raise ValueError("`temperature` must be positive. {}".format(temperature))

        if reduction not in {"none", "mean", "sum"}:
            raise ValueError('`reduction` must be in `{"none", "mean", "sum"}`. Not {}'.format(reduction))

        super(NT_Xent, self).__init__()
        self.cross_entropy = torch.nn.functional.cross_entropy
        self.temperature = temperature
        self.reduction = reduction
        self.device = device

    def forward(self, views: list) -> torch.FloatTensor:
        """
        Generalised version of SimCLR's InfoNCE loss.
        The number of augmentation can be larger than 2.

        :param views: list of feature representation. The shape is (T, N, D), where
            `T` is the number of views,
            `N` is the size of mini-batches (or the number of seed image, or K+1 in the paper),
            and `D` is the dimensionality of features.
        :return: Loss value. The shape depends on `reduction`: (2, N) or a scalar.
        """

        num_views = len(views)  # == T
        size_mini_batches = len(views[0])  # == N

        # normalisation
        views = [torch.nn.functional.normalize(view, p=2, dim=1) for view in views]  # T x N x D

        targets = torch.arange(size_mini_batches).to(self.device)  # == indices for positive pairs
        mask = ~torch.eye(size_mini_batches, dtype=torch.bool).to(self.device)  # to remove similarity to themselves

        entropy_losses = []
        for key_idx in range(num_views - 1):

            sim00 = torch.matmul(views[key_idx], views[key_idx].t()) / self.temperature  # N x N
            # remove own similarities
            sim00 = sim00[mask].view(size_mini_batches, -1)  # N x (N-1)

            for positive_view_idx in range(key_idx + 1, num_views):
                # negative
                sim11 = torch.matmul(views[positive_view_idx], views[positive_view_idx].t()) / self.temperature  # N x N
                # remove own similarities
                sim11 = sim11[mask].view(size_mini_batches, -1)  # N x (N-1)

                # positive and negatives
                sim01 = torch.matmul(views[key_idx], views[positive_view_idx].t()) / self.temperature  # N x N

                sim0 = [sim01, sim00]
                sim1 = [sim01.t(), sim11]

                exclude_indices = (key_idx, positive_view_idx)

                # other negatives
                for negative_views_idx in range(num_views):

                    if negative_views_idx in exclude_indices:
                        continue

                    neg = torch.matmul(views[key_idx], views[negative_views_idx].t()) / self.temperature  # N x N
                    sim0.append(neg[mask].view(size_mini_batches, -1))  # remove own similarity and add N x (N-1) tensor

                    neg = torch.matmul(views[positive_view_idx],
                                       views[negative_views_idx].t()) / self.temperature  # N x N
                    sim1.append(neg[mask].view(size_mini_batches, -1))  # remove own similarity and add N x (N-1) tensor

                sim0 = torch.cat(sim0, dim=1)  # N x (N + (#num-views-1) (N-1))
                sim1 = torch.cat(sim1, dim=1)  # N x (N + (#num-views-1) (N-1))

                if self.reduction == "none":
                    entropy_losses.append(self.cross_entropy(sim0, targets, reduction="none"))
                    entropy_losses.append(self.cross_entropy(sim1, targets, reduction="none"))
                else:
                    entropy_losses.append(
                        self.cross_entropy(sim0, targets, reduction="sum") + self.cross_entropy(sim1, targets,
                                                                                                reduction="sum"))

        if self.reduction == "none":
            return torch.stack(entropy_losses)
        if self.reduction == "sum":
            return torch.stack(entropy_losses).sum()  # shape: scalar
        else:
            return torch.stack(entropy_losses).sum() / 2. / size_mini_batches / len(entropy_losses)  # shape: scalar

    def simclr_forward_with_single_view_negative(self, views: list) -> torch.FloatTensor:
        """
        SimCLR's InfoNCE loss.
        It assumes the number of views is 2, and view0 is used for key,
        and view1 is used for positive / negative samples.

        :param views: list of feature representation. The shape is (2, N, D), where
            `T` is the number of views,
            N` is the size of mini-batches ( or the number of seed image) ,
            and `D` is the dimensionality of features.
        :return: Loss value. The shape depends on `reduction`: (N,) or a scalar.
        """
        size_mini_batches = len(views[0])  # == N

        targets = torch.arange(size_mini_batches).to(self.device)  # == indices for positive samples

        # normalise
        views = [torch.nn.functional.normalize(view, p=2, dim=1) for view in views]  # T x N x D

        cos = torch.matmul(views[0], views[1].t()) / self.temperature  # N x N
        return self.cross_entropy(cos, targets, reduction=self.reduction)

    def bound_loss(
            self, features: torch.FloatTensor, mu: torch.FloatTensor, targets: torch.LongTensor
    ) -> torch.FloatTensor:

        """
        :param features: Feature representation. The shape is (N, D),
            where `N` is the size of mini-batches (or the number of seed image),
            and `D` is the dimensionality of features.
        :param mu: Mean classifier's weights. shape: (C, D), where `C` is the number of classes.
        :param targets: Target classes.
        :return: loss function
        """

        unnormalized_scores = features.matmul(mu.t()) / self.temperature  # scaled cosine (N x C)
        return self.cross_entropy(unnormalized_scores, targets, reduction=self.reduction)
