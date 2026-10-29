import torch
import torch.nn as nn
import torch.nn.functional as F


class FeatureConditionedDecoderMultiTask(nn.Module):
    """
    Feature-conditioned multi-task decoder.

    The model receives two inputs:

        z:
            The weekly user embedding produced by your encoder.

        feature_ids:
           The feature IDs we want to check for that user-week.

    The model predicts two outputs:

        used_logit
           A raw logit for whether the feature was used.

        time_pred
           A non-negative prediction for elapsed_time.

    This is called "feature-conditioned" because the decoder does not directly
    output all features at once. Instead, it answers:

        "For this user-week embedding, what is the predicted usage
         for this specific feature?"

    So during inference, we loop over all feature IDs and ask the model
    to predict each one.
    """

    def __init__(
        self,
        z_dim: int,
        num_features: int,
        feat_emb_dim: int,
        hidden: int,
        dropout: float,
    ):
        """
        Initialize the decoder.

        Args:
            z_dim:
                Dimension of the weekly user embedding.
                Example: if your embedding columns are emb_0 to emb_191,
                then z_dim = 192.

            num_features:
                Total number of possible features.

            feat_emb_dim:
                Dimension of the learned feature embedding.

            hidden:
                Hidden dimension of the MLP.

            dropout:
                Dropout probability.
        """
        super().__init__()

        # ----------------------------------------------------
        # Learned feature embedding
        # ----------------------------------------------------
        #
        # Each feature_id gets its own learnable vector.
        #
        # Example:
        #   feature_id 0 -> vector
        #   feature_id 1 -> vector
        #   feature_id 2 -> vector
        #
        # This lets the model learn that different features behave differently.
        self.feat_emb = nn.Embedding(
            num_embeddings=num_features,
            embedding_dim=feat_emb_dim,
        )

        # ----------------------------------------------------
        # Feature-specific classification bias
        # ----------------------------------------------------
        #
        # Some features are naturally more common than others.
        #
        # This bias gives each feature its own baseline tendency to be predicted
        # as used or not used.
        self.feature_bias = nn.Embedding(
            num_embeddings=num_features,
            embedding_dim=1,
        )

        nn.init.zeros_(self.feature_bias.weight)

        # ----------------------------------------------------
        # Shared MLP backbone
        # ----------------------------------------------------
        #
        # Input to the model is:
        #
        #   [user-week embedding, feature embedding]
        #
        # If:
        #   z_dim = 192
        #   feat_emb_dim = 256
        #
        # then the input size is:
        #   448
        self.backbone = nn.Sequential(
            nn.Linear(z_dim + feat_emb_dim, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # ----------------------------------------------------
        # Head 1: used / not-used classification
        # ----------------------------------------------------
        #
        # This predicts one raw logit.
        #
        # Later:
        #   probability = sigmoid(used_logit)
        self.used_head = nn.Linear(hidden, 1)

        # ----------------------------------------------------
        # Head 2: elapsed-time regression
        # ----------------------------------------------------
        #
        # This predicts the amount of elapsed_time for the feature.
        self.time_head = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),

            nn.Linear(hidden // 2, 1),
        )

    def forward(
        self,
        z: torch.Tensor,
        feature_ids: torch.Tensor,
    ):
        """
        Forward pass.

        Args:
            z:
                Tensor of user-week embeddings.

                Shape:
                    (batch_size, z_dim)

            feature_ids:
                Tensor of feature IDs.

                Shape:
                    (batch_size,)

        Returns:
            used_logit:
                Raw classification logits.

                Shape:
                    (batch_size,)

            time_pred:
                Non-negative elapsed-time predictions.

                Shape:
                    (batch_size,)
        """

        # Convert feature IDs into learned feature vectors.
        feature_embedding = self.feat_emb(feature_ids)

        # Combine the user-week embedding and the feature embedding.
        x = torch.cat([z, feature_embedding], dim=1)

        # Shared representation.
        h = self.backbone(x)

        # Classification output.
        used_logit = self.used_head(h).squeeze(1)

        # Add feature-specific bias.
        used_logit = used_logit + self.feature_bias(feature_ids).squeeze(1)

        # Regression output.
        time_raw = self.time_head(h).squeeze(1)

        # softplus makes sure elapsed-time prediction is non-negative.
        time_pred = F.softplus(time_raw)

        return used_logit, time_pred