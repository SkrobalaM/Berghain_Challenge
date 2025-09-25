import math
import random

def bernoulli_joint(pY, pW, rho):
    d = rho * math.sqrt(pY*(1-pY)*pW*(1-pW))
    p11 = pY*pW + d
    p10 = pY - p11
    p01 = pW - p11
    p00 = 1 - p11 - p10 - p01
    return tuple(max(0.0, min(1.0, x)) for x in (p11,p10,p01,p00))

def sample_person(pY, pW, rho):
    p11,p10,p01,p00 = bernoulli_joint(pY,pW,rho)
    u = random.random()
    if u < p11: return "11"
    u -= p11
    if u < p10: return "10"
    u -= p10
    if u < p01: return "01"
    return "00"

def select(p,marge):
	if int(p[0]) == 1 or int(p[1])==1:
		return True
	else:
		x = random.random()
		if x < marge:
			return True
		else:
			return False

def finised(berg,min_cap):
	result = True
	for trait in berg:
		if berg[trait] < min_cap:
			result = False
	return result


def simulate():
	N, max_cap, y_min, w_min = 1000, 21000, 600, 600
	pY, pW, rho = 0.3225, 0.3225, 0.18304299322062992
	marge = 0.1

	accepted = 0
	refused = 0
	count = 0
	berg={"young": 0, "well-dressed":0}
	display = False

	while accepted < 1000 and count <max_cap:
		person = sample_person(pY,pW,rho)
		if select(person,marge) or finised(berg,y_min):
			if int(person[0])==1:
				berg["young"] +=1
			if int(person[1])==1:
				berg["well-dressed"] +=1
			accepted += 1
		else:
			refused +=1
		count += 1
	success = finised(berg,y_min)
	if display:
		print("Stats :",berg," /accepted :",accepted," /refused :",refused, " /sucess :",success)
	return success,refused

success_tot=0
failure_tot=0
refused_tot=0
best_refused=21000


for i in range(10000):
	if (i%1000) == 0:
		print("Sim :",i)
	success_res,refused=simulate()
	if success_res:
		success_tot+=1
		refused_tot += refused
		if refused<best_refused:
			best_refused = refused
	else: 
		failure_tot+=1


print(	"Failure :", failure_tot," /success :",success_tot," /Ratio :",(failure_tot/success_tot), 
		" /Avg rejected :", refused_tot/success_tot, " /Best refused :", best_refused)






