#This model uses attention from user to review to resturaunt
#       From the attention, it determines the credibility weights for aggregating review signals

import torch, torch.nn as nn

class ReviewCredibilityGAT(nn.Module):
    def __init__(self, in_dim, hidden=32, dropout=0.1):
        super().__init__()
        #query - asking info
        self.q = nn.Linear(in_dim, hidden)
        #key - descriptor possible source
        self.k = nn.Linear(in_dim, hidden)
        #value - actual content
        self.v = nn.Linear(in_dim, 1)

        self.dropout = nn.Dropout(dropout)

    def forward(self, review_x, review_to_rest_idx):
        #Simple attention: score = q*k, then values = v(review)
        #use v to represent indv vec
        qv = self.q(review_x); kv = self.k(review_x)
        att = (qv * kv).sum(-1)     #[num_reviews]
        #groupwise softmax per restaurant
        weights = torch.zeros_like(att)
        for r in torch.unique(review_to_rest_idx):
            mask = (review_to_rest_idx == r)
            w = torch.softmax(att[mask], dim=0)
            weights[mask] = w
        return weights      #per-review cred
