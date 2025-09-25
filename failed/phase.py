import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib
import random
matplotlib.use("TkAgg")

def generate_person(p,corr):
    thresholds = [norm.ppf(v) for k, v in p.items()]
    sample = np.random.multivariate_normal([0, 0, 0, 0], corr)
    person = {"techno_lover":0,"well_connected":0,"creative":0,"berlin_local":0}
    for i,trait in enumerate(person.keys()):
        if sample[i] < thresholds[i]:
            person[trait] = True
        else:
            person[trait] = False

    return person

def build_key(person):
    key=""
    for trait in person:
        if person[trait]==True:
            key+="1"
        else:
            key+="0"
    return key

def label_from_bits(bits):
    traits = ["techno_lover", "well_connected", "creative", "berlin_local"]
    present = [t for bit, t in zip(bits, traits) if bit == '1']
    return ", ".join(present) if present else "none"

def build_graph(data):
    traits = ["techno_lover", "well_connected", "creative", "berlin_local"]
    rows = []
    for bits, count in data.items():
        label = label_from_bits(bits)
        rows.append({"trait_group": label, "occurrence": count})

    df = pd.DataFrame(rows)

    df_sorted = df.sort_values("occurrence", ascending=False)

    # Plot the bar chart
    plt.figure(figsize=(12, 6))
    plt.bar(df_sorted["trait_group"], df_sorted["occurrence"])
    plt.xlabel("Trait group (present traits)")
    plt.ylabel("Occurrence")
    plt.title("Occurrences by Trait Group")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.show()


def finished(berg,min_cap,done):
    for trait in berg:
        if berg[trait] >= min_cap[trait]:
            done[trait] = True
    return done


def count_trait(p):
    result = 0
    for trait in p:
        if p[trait]==True:
            result +=1
    return result

def count_finished(done):
    result = 0
    for trait in done:
        if done[trait]==True:
            result +=1
    return result

def select(p,done,phase,accepted):
    global first1
    global first2
    global first3
    global first4
    display = True

    result = False
    if display:
        if phase == 1:
            if first1:
                print("Phase 1 :",accepted,done)
                first1 = False
        elif phase == 2:
            if first2:
                print("Phase 2 :",accepted,done)
                first2 = False
        elif phase == 3:
            if first3:
                print("Phase 3 :",accepted,done)
                first3 = False
        elif phase == 4:
            if first4:
                print("Phase 4 :",accepted,done)
                first4 = False

    if phase == 0:
        for trait in done:
            if done[trait]==False:
                if (p[trait]==True and count_trait(p) >= 3) or (p["creative"]==True and count_trait(p) >= 2):
                    result = True


    elif phase == 1:
        for trait in done:
            if done[trait]==False:
                if (p[trait]==True and count_trait(p) >= 3) or (p["creative"]==True and count_trait(p) >= 2):
                    result = True
        x =random.random()
        if x < 0.04:
            result = True

    elif phase == 2:
        for trait in done:
            if done[trait]==False:
                if p[trait]==True and count_trait(p) >= 2 or (p["berlin_local"]==True):
                    result = True
        x =random.random()
        if x < 0.08:
            result = True

    elif phase == 3:
        for trait in done:
            if done[trait]==False:
                if p[trait]==True:
                    result = True
        x =random.random()
        if x < 0.17:
            result = True

    elif phase == 4:
        result = True



    return result


min_cap = {
    "techno_lover": 650,
    "well_connected": 450,
    "creative": 300,
    "berlin_local": 750}

prob = {
    "techno_lover": 0.6265,
    "well_connected": 0.47,
    "creative": 0.06227,
    "berlin_local": 0.398}

corr = np.array([
        [1, -0.4696169332674324, 0.09463317039891586, -0.6549403815606182],
        [-0.4696169332674324, 1, 0.14197259140471485, 0.5724067808436452],
        [0.09463317039891586, 0.14197259140471485, 1, 0.14446459505650772],
        [-0.6549403815606182, 0.5724067808436452, 0.14446459505650772, 1]
        ])

max_p=21000

build = False

first1 = True
first2 = True
first3 = True
first4 = True
distribution={
            "0000":0,"0001":0,"0010":0,"0011":0,
            "0100":0,"0101":0,"0110":0,"0111":0,
            "1000":0,"1001":0,"1010":0,"1011":0,
            "1100":0,"1101":0,"1110":0,"1111":0,
            }




def simulate(prob,corr,max_p,build=False):
    berg = {"techno_lover":0,"well_connected":0,"creative":0,"berlin_local":0}
    done = {"techno_lover":False,"well_connected":False,"creative":False,"berlin_local":False}
    rejected = 0
    accepted = 0
    count = 0
    display = True
    while accepted < 1000 and count < max_p:


        p=generate_person(prob,corr)
        done = finished(berg,min_cap,done)
        if select(p,done,count_finished(done),accepted):
            for trait in p:
                if p[trait] == True:
                    berg[trait]+=1
            accepted +=1
        else:
            rejected+=1
        count +=1

        if build:
            key = build_key(p)
            distribution[key]+=1
    if build:
        build_graph(distribution)
    succes = False
    if count_finished(done) == 4:
        succes = True
    if display:
        print("Done :",done)

        print("Stats :",berg," /accepted :",accepted, " /rejected :",rejected, " /succes :", succes)
    return berg,done,accepted,rejected,succes


succes = 0
failure = 0
rejected_tot = 0
sim =0
simulate(prob,corr,max_p,build)










