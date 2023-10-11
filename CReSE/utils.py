import torch.nn.functional as F

import matplotlib.pyplot as plt



class AvgMeter:
    def __init__(self, name="Metric"):
        self.name = name
        self.reset()

    def reset(self):
        self.avg, self.sum, self.count = [0] * 3

    def update(self, val, count=1):
        self.count += count
        self.sum += val * count
        self.avg = self.sum / self.count
        
    def __repr__(self):
        title = f"{self.name}: {self.avg:.4f}"
        return title
    
    
def calculate_match_score(model, ec1, ec2):
    ec1_embedding = model.get_ec_embeddings(ec1)
    ec2_embedding = model.get_ec_embeddings(ec2)
    
    #normalized and measure dot distance
    ec1_embedding = F.normalize(ec1_embedding, p=2, dim=-1)
    ec2_embedding = F.normalize(ec2_embedding, p=2, dim=-1)
    match_score = ec1_embedding @ ec2_embedding.T
    
    return match_score


def get_match_scores(model, ECs_cr_triplets):
    match_scores = []
    for ec1, ec2, _ in ECs_cr_triplets:
        match_score = calculate_match_score(model, ec1, ec2)
        match_scores.append(match_score.tolist()[0][0])
        
    return match_scores
    
    
def min_max_scaling(values):
    min_value = min(values); max_value = max(values)
    return [(value - min_value) / (max_value - min_value) for value in values]
    
    
#print-out graphs
def draw_scatter_plot_mss_crs(match_scores, crs):
    #draw scatter plot
    fig, ax = plt.subplots()
    ax.scatter(match_scores, crs)
    ax.set_xlabel('Match scores by EC-EC CLIP model')
    ax.set_ylabel('Clinical relevance asses by ChatGPT')
    ax.set_xlim([-0.1, 1.1])
    ax.set_yticks(range(0, 4, 1))
    plt.show()
    
    
def draw_scatter_plot_mss_crs2(match_scores, crs):
    #draw scatter plot
    fig, ax = plt.subplots()
    ax.scatter(match_scores, crs)
    ax.set_xlabel('Match scores by EC-EC CLIP model')
    ax.set_ylabel('Clinical relevance asses by ChatGPT')
    ax.set_yticks(range(0, 4, 1))
    plt.show()


def draw_epoch_graph(list1, list2, lable1, label2):
    fig, ax = plt.subplots()
    ax.plot(list(range(1, len(list1)+1)), list1, label=lable1)
    ax.plot(list(range(1, len(list2)+1)), list2, label=label2)
    ax.legend()
    ax.set_xlabel(f'Epoch')
    ax.set_ylabel(f'{lable1} & {label2}')
