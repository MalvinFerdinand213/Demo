import torch
import torch.nn as nn
from PIL import Image

ASPECTS = ["Cleanliness", "Room", "Service / Staff", "Location", "Facilities", "Food & Beverage"]

class MiniMultimodalAspectModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.text_fc = nn.Linear(100, 32)
        self.img_fc = nn.Linear(100, 32)
        self.aspect_fc = nn.Linear(64, len(ASPECTS))
        self.sigmoid = nn.Sigmoid()

    def forward(self, text_dummy, img_dummy):
        t_feat = self.text_fc(text_dummy)
        i_feat = self.img_fc(img_dummy)
        fused = torch.cat((t_feat, i_feat), dim=-1)
        scores = self.sigmoid(self.aspect_fc(fused))
        return scores

def predict_aspect_scores(review_text: str, image: Image.Image, star_rating: int):
    torch.manual_seed(len(review_text) + star_rating)
    t_dummy = torch.randn(1, 100)
    i_dummy = torch.randn(1, 100)
    
    model = MiniMultimodalAspectModel()
    model.eval()
    
    with torch.no_grad():
        raw_scores = model(t_dummy, i_dummy).squeeze().tolist()
    
    results = {}
    for idx, aspect in enumerate(ASPECTS):
        score = round(raw_scores[idx] * 5, 2)
        if score >= 3.5:
            polarity = "Positive"
        elif score >= 2.5:
            polarity = "Neutral"
        else:
            polarity = "Negative"
            
        results[aspect] = {
            "score": score,
            "polarity": polarity
        }
    return results

def get_star_tier_benchmark(star_rating: int):
    base = star_rating * 0.8
    benchmarks = {
        "Cleanliness": round(min(5.0, base + 0.5), 2),
        "Room": round(min(5.0, base + 0.3), 2),
        "Service / Staff": round(min(5.0, base + 0.4), 2),
        "Location": round(min(5.0, base + 0.6), 2),
        "Facilities": round(min(5.0, base + 0.2), 2),
        "Food & Beverage": round(min(5.0, base + 0.3), 2),
    }
    return benchmarks
