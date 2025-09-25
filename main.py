from __future__ import annotations
import itertools, hashlib
import numpy as np
from scipy.stats import norm, multivariate_normal
from scipy.optimize import brentq, linprog
import random
import requests
import threading
from concurrent.futures import ProcessPoolExecutor, as_completed
import time
from urllib.parse import urlencode
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

playerId="**********************"

seed=random.randrange(1,100000)
rng = np.random.default_rng(seed)

data1 = { 
	"constraints":[
		{"attribute":"young","minCount":600},
		{"attribute":"well_dressed","minCount":600}
	],
 	"attributeStatistics":{
 		"relativeFrequencies":{
 			"well_dressed":0.3225,
		 	"young":0.3225
		},
	 	"correlations":{
	 		"well_dressed":{"well_dressed":1,"young":0.18304299322062992},
		 	"young":{"well_dressed":0.18304299322062992,"young":1}
		 }
	}
}

data2 = {
	"constraints": [
		{"attribute":"techno_lover","minCount":650},
		{"attribute":"well_connected","minCount":450},
		{"attribute":"creative","minCount":300},
		{"attribute":"berlin_local","minCount":750}],
	"attributeStatistics": {
		"relativeFrequencies": {
			"techno_lover":0.6265,
			"well_connected":0.4700,
			"creative":0.06227,
			"berlin_local":0.3980,},
		"correlations": {
			"techno_lover":{"techno_lover":1,"well_connected":-0.4696169332674324,"creative":0.09463317039891586,"berlin_local":-0.6549403815606182},
			"well_connected":{"techno_lover":-0.4696169332674324,"well_connected":1,"creative":0.14197259140471485,"berlin_local":0.5724067808436452},
			"creative":{"techno_lover":0.09463317039891586,"well_connected":0.14197259140471485,"creative":1,"berlin_local":0.14446459505650772},
			"berlin_local":{"techno_lover":-0.6549403815606182,"well_connected":0.5724067808436452,"creative":0.14446459505650772,"berlin_local":1}}}}

data3 = {"constraints":[
			{"attribute":"underground_veteran","minCount":500},
			{"attribute":"international","minCount":650},
			{"attribute":"fashion_forward","minCount":550},
			{"attribute":"queer_friendly","minCount":250},
			{"attribute":"vinyl_collector","minCount":200},
			{"attribute":"german_speaker","minCount":800}],
		"attributeStatistics":{
			"relativeFrequencies":{
				"underground_veteran":0.6794999999999999,
				"international":0.5735,
				"fashion_forward":0.6910000000000002,
				"queer_friendly":0.04614,
				"vinyl_collector":0.044539999999999996,
				"german_speaker":0.4565000000000001},
			"correlations":{
				"underground_veteran":{"underground_veteran":1,"international":-0.08110175777152992,"fashion_forward":-0.1696563475505309,"queer_friendly":0.03719928376753885,"vinyl_collector":0.07223521156389842,"german_speaker":0.11188766703422799},
				"international":{"underground_veteran":-0.08110175777152992,"international":1,"fashion_forward":0.375711059360155,"queer_friendly":0.0036693314388711686,"vinyl_collector":-0.03083247098181075,"german_speaker":-0.7172529382519395},
				"fashion_forward":{"underground_veteran":-0.1696563475505309,"international":0.375711059360155,"fashion_forward":1,"queer_friendly":-0.0034530926793377476,"vinyl_collector":-0.11024719606358546,"german_speaker":-0.3521024461597403},
				"queer_friendly":{"underground_veteran":0.03719928376753885,"international":0.0036693314388711686,"fashion_forward":-0.0034530926793377476,"queer_friendly":1,"vinyl_collector":0.47990640803167306,"german_speaker":0.04797381132680503},
				"vinyl_collector":{"underground_veteran":0.07223521156389842,"international":-0.03083247098181075,"fashion_forward":-0.11024719606358546,"queer_friendly":0.47990640803167306,"vinyl_collector":1,"german_speaker":0.09984452286269897},
				"german_speaker":{"underground_veteran":0.11188766703422799,"international":-0.7172529382519395,"fashion_forward":-0.3521024461597403,"queer_friendly":0.04797381132680503,"vinyl_collector":0.09984452286269897,"german_speaker":1}}}}
data=[data1,data2,data3]

class Senario:

	def __init__(self, data):
		self.gameId = data["gameId"]
		self.constraints = data["constraints"]
		self.relativeFrequencies = data["attributeStatistics"]["relativeFrequencies"]
		self.correlations = data["attributeStatistics"]["correlations"]


def start_senario(n,playerId):
	url = "https://berghain.challenges.listenlabs.ai/new-game?scenario="+str(n)+"&"+playerId+"=19e0bae1-c2ca-428f-a9c9-d2b3160d52ee"
	try:
		response = requests.get(url)

		if response.status_code == 200:
			data = response.json()
			print("Successfully fetched data:")
			return data
		else:
			print(f"Error: {response.status_code} - {response.text}")
			return None

	except requests.exceptions.RequestException as e:
		print(f"An error occurred during the request: {e}")
		return None



def make_session(total_retries=5, backoff_factor=0.5):
	retry = Retry(
		total=total_retries,
		connect=total_retries,
		read=total_retries,
		status=total_retries,
		backoff_factor=backoff_factor,
		status_forcelist=(429, 500, 502, 503, 504),
		allowed_methods=frozenset(["GET", "POST", "PUT", "DELETE", "HEAD", "OPTIONS"]),
		raise_on_status=False,
	)
	adapter = HTTPAdapter(max_retries=retry, pool_connections=10, pool_maxsize=10)
	s = requests.Session()
	s.mount("https://", adapter)
	s.mount("http://", adapter)
	s.headers.update({
		"User-Agent": "python-requests/visio-client",
		"Connection": "close"
	})
	return s




def visio(game_id, index, accept, timeout=(5, 10)):
	SESSION = make_session()
	BASE = "https://berghain.challenges.listenlabs.ai"
	params = {
		"gameId": game_id,
		"personIndex": int(index),
		"accept": str(accept).lower() if isinstance(accept, bool) else str(accept),
	}
	url = f"{BASE}/decide-and-next?{urlencode(params)}"
	try:
		resp = SESSION.get(url, timeout=timeout)
		if resp.status_code == 200:
			return resp.json()
		else:
			print(f"HTTP {resp.status_code}: {resp.text[:200]}")
			return None
	except requests.exceptions.SSLError as e:
		print(f"SSL error: {e}")
		return None
	except requests.exceptions.RequestException as e:
		print(f"Request error: {e}")
		return None

def ask_api(game_id, index, accept, max_attempts=6, base_sleep=0.5):
	attempt = 1
	while attempt <= max_attempts:
		data = visio(game_id, index, accept)
		if data is not None:
			return data
		sleep = base_sleep * (2 ** (attempt - 1))
		time.sleep(sleep + 0.1 * (attempt % 3))
		attempt += 1
	return None






def bernoulli_phi_bounds(p, q):
	denom = np.sqrt(p*(1-p)*q*(1-q))
	if denom == 0.0:
		return (0.0, 0.0)
	qmin = max(0.0, p+q-1.0)
	qmax = min(p, q)
	return ((qmin - p*q)/denom, (qmax - p*q)/denom)

def phi_from_latent_rho(rho, p1, p2):
	t1 = norm.ppf(1-p1)
	t2 = norm.ppf(1-p2)
	cov = np.array([[1.0, rho],[rho,1.0]])
	q_le = multivariate_normal.cdf([t1,t2], mean=[0,0], cov=cov)
	q11 = 1 - norm.cdf(t1) - norm.cdf(t2) + q_le
	denom = np.sqrt(p1*(1-p1)*p2*(1-p2))
	return 0.0 if denom==0 else (q11 - p1*p2)/denom

def invert_phi_to_rho(phi_target, p1, p2):
	lo, hi = -0.999, 0.999
	phi_min, phi_max = bernoulli_phi_bounds(p1, p2)
	pt = min(max(phi_target, phi_min), phi_max)
	g = lambda r: phi_from_latent_rho(r, p1, p2) - pt
	glo, ghi = g(lo), g(hi)
	if glo==0: return lo
	if ghi==0: return hi
	if glo*ghi > 0:
		xs = np.linspace(lo, hi, 401)
		ys = [g(x) for x in xs]
		return xs[int(np.argmin(np.abs(ys)))]
	return brentq(g, lo, hi, maxiter=200, xtol=1e-6)

def nearest_psd(A, eps=1e-8):
	B = (A + A.T)/2.0
	w, V = np.linalg.eigh(B)
	w = np.clip(w, eps, None)
	return (V * w) @ V.T

def estimate_joint_by_gaussian_copula(p_vec, Phi, n_samples=200_000, seed=123):
	n = len(p_vec)
	R = np.eye(n)
	for i in range(n):
		for j in range(i+1, n):
			R[i,j] = R[j,i] = invert_phi_to_rho(Phi[i,j], p_vec[i], p_vec[j])
	R = nearest_psd(R, eps=1e-8)
	t = norm.ppf(1-np.array(p_vec))
	rng = np.random.default_rng(seed)
	L = np.linalg.cholesky(R + 1e-12*np.eye(n))
	Z = rng.standard_normal(size=(n_samples, n)) @ L.T
	X = (Z > t).astype(np.int8)
	idx = (X * (1 << np.arange(n))).sum(axis=1)
	counts = np.bincount(idx, minlength=1<<n)
	pi = counts / counts.sum()
	return pi, R

def state_bits(idx, n):
	return tuple((idx >> k) & 1 for k in range(n))

def pretty_state(idx):
	bits = state_bits(idx, n)
	return "{" + ", ".join(f"{a}:{b}" for a,b in zip(ATTRS,bits)) + "}"


def generate_person(p_vec, latent_corr, rng):
	t = norm.ppf(1 - np.array(p_vec))
	z = rng.multivariate_normal(np.zeros(len(p_vec)), latent_corr)
	x = (z > t).astype(int)  # 1 means attribute present
	return dict(zip(ATTRS, x.astype(bool)))



def count_finished(done):
	result = 0
	for trait in done:
		if done[trait]==True:
			result +=1
	return result

def build_bits(person):
	bits = [0]*len(person)
	idx = 0
	for trait in person:
		if person[trait]== True:
			bits[idx]=1
		idx+=1
	return tuple(bits)

def indicator(indicator_data, bits, rng,done):
	p = indicator_data[bits]
	return rng.random() < p

def finished(berg,min_cap,done):
	for trait in berg:
		if berg[trait] >= min_cap[trait]:
			done[trait] = True
	return done


def build_indicator(ATTRS, pi, targets_remaining, slots_remaining):
	theta = np.array([targets_remaining[a]/slots_remaining for a in ATTRS])
	n = len(ATTRS)
	n_states = 1 << n
	ones = np.ones(n_states)

	A_ub = []
	b_ub = []
	for j in range(n):
		mask_j = np.array([(s >> j) & 1 for s in range(n_states)], dtype=float)
		A_ub.append(-mask_j + theta[j]*ones)
		b_ub.append(0.0)
	A_ub = np.array(A_ub); b_ub = np.array(b_ub)

	res = linprog(-np.ones(n_states),
				  A_ub=A_ub, b_ub=b_ub,
				  bounds=[(0.0, float(pi_s)) for pi_s in pi],
				  method="highs")
	if not res.success:
		raise RuntimeError("LP infeasible for current deficits; check r and d_j.")

	y = res.x
	accept_prob = np.divide(y, pi, out=np.zeros_like(y), where=pi>1e-15)
	accept_prob = np.clip(accept_prob, 0.0, 1.0)
	indicator_data={}
	for idx, p_s in enumerate(accept_prob):
		indicator_data[state_bits(idx, n)]=float(p_s)
	return indicator_data


def build_targets(berg,targets):
	new_targets = {"techno_lover":0,"well_connected":0,"creative":0,"berlin_local":0}
	for trait in berg:
		new_targets[trait] = max(targets[trait]-berg[trait],0)
	return new_targets

def simulate():
	global prob,latent_corr,rng,targets,ATTRS,pi
	berg = {"techno_lover":0,"well_connected":0,"creative":0,"berlin_local":0}
	done = {"techno_lover":False,"well_connected":False,"creative":False}
	rejected = 0
	accepted = 0
	count = 0
	display = False
	max_p = 21000
	max_cap=1000
	hypothetic_cap=1000
	new_targets = build_targets(berg,targets)
	indicator_data=build_indicator(ATTRS,pi,new_targets,hypothetic_cap-accepted)

	while accepted < max_cap and count < max_p:
		person = generate_person(prob, latent_corr, rng)
		bits = build_bits(person)
		if indicator(indicator_data, bits, rng,done):
			for trait, val in person.items():
				if val:
					berg[trait] += 1
			accepted += 1
			if accepted <max_cap:
				new_targets = build_targets(berg,targets)
				indicator_data=build_indicator(ATTRS,pi,new_targets,hypothetic_cap-accepted)
		else:
			rejected += 1
		count += 1
		
		done = finished(berg,targets,done)
	succes = False
	if count_finished(done) == len(berg):
		succes = True
	if display:
		print("Done :",done)

		print("Stats :",berg," /accepted :",accepted, " /rejected :",rejected, " /succes :", succes)

	return succes,rejected



def launch_senario(count_T,senario,data):
	s = Senario(start_senario(senario,playerId))
	gameID = s.gameId
	seed = random.randrange(1,100000)
	ATTRS = []
	berg={}
	done={}
	for trait in data["attributeStatistics"]["relativeFrequencies"]:
		ATTRS.append(trait)
		berg[trait]=0
		done[trait]=False
	prob = np.array([data["attributeStatistics"]["relativeFrequencies"][a] for a in ATTRS])
	Phi = np.array([[data["attributeStatistics"]["correlations"][ai][aj] for aj in ATTRS] for ai in ATTRS])
	targets = {c["attribute"]: c["minCount"] for c in data["constraints"]}
	rng = np.random.default_rng(seed)
	pi, latent_corr = estimate_joint_by_gaussian_copula(prob, Phi, n_samples=200_000, seed=seed)


	rejected = 0
	accepted = 0
	count = 0
	display = True
	max_p = 21000
	max_cap=1000
	hypothetic_cap=1000
	new_targets = build_targets(berg,targets)
	indicator_data=build_indicator(ATTRS,pi,new_targets,hypothetic_cap-accepted)

	data = ask_api(gameID,count,"false")

	while accepted < max_cap and count < max_p:
		if (count%200==0) and display:
			print("Thread :",count_T," /Stats :",berg," /accepted :",accepted, " /rejected :",rejected)
		count += 1
		person = data["nextPerson"]["attributes"]
		bits = build_bits(person)
		if indicator(indicator_data, bits, rng,done):
			for trait, val in person.items():
				if val:
					berg[trait] += 1
			accepted += 1
			if accepted < max_cap:
				new_targets = build_targets(berg,targets)
				indicator_data=build_indicator(ATTRS,pi,new_targets,max_cap-accepted)
			data = ask_api(gameID,count,"true")
		else:
			rejected += 1
			data = ask_api(gameID,count,"false")
		done = finished(berg,targets,done)
	succes = False
	if count_finished(done) == len(berg):
		succes = True
	if display:
		print("Thread :",count_T,"Done :",done)

		print("Senario :",senario ," /thread :",count_T," /Stats :",berg," /accepted :",accepted, " /rejected :",rejected, " /succes :", succes)
	return succes

def test():
	thread = threading.Thread(target=launch_senario, args=(1,2,data2))
	thread.start()
	thread.join()

def loop():
	count = 0
	senario = 0
	while True:
		for i in range(10):
			senario = ((senario) % 2)+1
			thread = threading.Thread(target=launch_senario, args=(count,2,data[1]))
			thread.start()
			
			count+=1
		time.sleep(960)


loop()
