import numpy as np
from scipy.stats import norm

def generate_person(p,corr):
    thresholds = {k: norm.ppf(v) for k, v in p.items()}
    sample = np.random.multivariate_normal([0, 0, 0, 0], corr)
    person = {}
    for i, trait in enumerate(p.keys()):
        person[trait] = sample[i] < thresholds[trait]

    return person

def compute_rarity(trait_l,prob_l,max_cap):
    rarity = {"techno_lover":0,"well_connected":0,"creative":0,"berlin_local":0}
    for key in (rarity):
        rarity[key] = (max_cap[key]-trait_l[key])/prob_l[key]**(1.5)
    return rarity


def compute_score(p,rarity):
    score = 0
    for key in (p):
        if (p[key]==True):
            score += rarity[key]
    return score





max_cap = {
    "techno_lover": 650,
    "well_connected": 450,
    "creative": 300,
    "berlin_local": 750}

prob_l = {
    "techno_lover": 0.6265,
    "well_connected": 0.47,
    "creative": 0.06227,
    "berlin_local": 0.398}

correlation = np.array([
        [1, -0.4696169332674324, 0.09463317039891586, -0.6549403815606182],
        [-0.4696169332674324, 1, 0.14197259140471485, 0.5724067808436452],
        [0.09463317039891586, 0.14197259140471485, 1, 0.14446459505650772],
        [-0.6549403815606182, 0.5724067808436452, 0.14446459505650772, 1]
        ])


trait_l={"techno_lover":0,"well_connected":0,"creative":0,"berlin_local":0}
max_p=21000
thresh = 1200
refused = 0
accepted = 0
total_score=0
max_score=0
min_score = 1000000

for i in range(max_p):
    p=generate_person(prob_l,correlation)
    rarity = compute_rarity(trait_l,prob_l,max_cap)
    score = compute_score(p,rarity)
    total_score+=score
    if score > max_score:
        max_score=score

    if score < min_score:
        min_score=score
    
    if score > thresh:
        accepted+=1
        for key in (trait_l):
            if (p[key]) == True:
                trait_l[key]+=1

    else:
        refused += 1
    
    finished = True
    for key in (trait_l):
        if trait_l[key]<max_cap[key]:
            finished = False

    if (finished) or (accepted==1000):
        break
print(total_score/max_p,"/",max_score,"/",min_score)
print("Stats: ",trait_l,"/total refused :",refused,"/accepted :",accepted,"/sucess :",finished)




