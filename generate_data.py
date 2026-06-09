import pandas as pd, numpy as np, math, json, os, urllib.request, ssl
from collections import defaultdict
from datetime import date, timedelta

HOY = date.today()
ssl._create_default_https_context = ssl._create_unverified_context

# Descargar/actualizar dataset
DATA_URL = "https://raw.githubusercontent.com/martj42/international_results/master/results.csv"
if not os.path.exists("internationals.csv"):
    urllib.request.urlretrieve(DATA_URL, "internationals.csv")

df = pd.read_csv("internationals.csv", usecols=['date','home_team','away_team','home_score','away_score'])
df = df.dropna(subset=['home_score','away_score'])
df['date'] = pd.to_datetime(df['date'], errors='coerce')
df = df.dropna(subset=['date']).sort_values('date').tail(2000).copy()

# Elo y modelo
K=20; elo=defaultdict(lambda:1500)
X,y=[],[]
for _,row in df.iterrows():
    h,a,gh,ga = row['home_team'],row['away_team'],row['home_score'],row['away_score']
    Rh,Ra=elo[h],elo[a]; avg=sum(elo.values())/len(elo) if elo else 1500
    neutral = not (h in ["Mexico","Canada","United States"] or a in ["Mexico","Canada","United States"])
    bh,ba=(1.2,1.2) if neutral else (1.4,1.1)
    fh=math.exp((Rh-avg)/400); fa=math.exp((Ra-avg)/400)
    lh=max(0.3,min(4.0,bh*fh/(fa**0.3))); la=max(0.3,min(4.0,ba*fa/(fh**0.3)))
    win=0
    for i in range(9):
        for j in range(9):
            p = (lh**i*math.exp(-lh)/math.factorial(i))*(la**j*math.exp(-la)/math.factorial(j))
            if i>j: win+=p
    X.append([Rh,Ra,int(not neutral),Rh-Ra])
    y.append(1 if gh>ga else 0)
    Eh=1/(1+10**((Ra-Rh)/400))
    if gh>ga: Sh,Sa=1,0
    elif gh==ga: Sh,Sa=0.5,0.5
    else: Sh,Sa=0,1
    elo[h]=Rh+K*(Sh-Eh); elo[a]=Ra+K*(Sa-(1-Eh))

from sklearn.linear_model import LogisticRegression
model = LogisticRegression(solver='lbfgs')
model.fit(X,y)

# Calendario completo de grupos (72 partidos)
grupos = {
    "A": ["Mexico","Canada","France","Peru"], "B": ["Brazil","Serbia","England","Iran"],
    "C": ["Argentina","Egypt","Spain","Australia"], "D": ["Germany","Morocco","Netherlands","Japan"],
    "E": ["Uruguay","South Korea","Belgium","Panama"], "F": ["Portugal","Ghana","United States","Switzerland"],
    "G": ["Italy","Nigeria","Colombia","Saudi Arabia"], "H": ["Senegal","Denmark","Chile","Qatar"],
    "I": ["Croatia","Russia","Cameroon","Jamaica"], "J": ["Spain","Scotland","Norway","Costa Rica"],
    "K": ["Belgium","Morocco","Canada","Panama"], "L": ["Portugal","Uruguay","South Korea","Ghana"],
}
enfrentamientos = [(0,1),(2,3),(0,2),(1,3),(0,3),(1,2)]
partidos_calendario = {}
inicio = date(2026,6,11)
for g, eqs in grupos.items():
    for i,(e1,e2) in enumerate(enfrentamientos):
        dia = inicio + timedelta(days=i//4)
        key = dia.isoformat()
        if key not in partidos_calendario:
            partidos_calendario[key] = []
        partidos_calendario[key].append({
            "local": eqs[e1], "visitante": eqs[e2], "grupo": g,
            "hora": f"{12+(i%4)}:00"
        })

# Predecir todos los partidos
data = {}
for fecha_str, partidos in partidos_calendario.items():
    data[fecha_str] = []
    for p in partidos:
        h,a = p["local"], p["visitante"]
        elo_h = elo.get(h,1500); elo_a = elo.get(a,1500)
        neutral = not (h in ["Mexico","Canada","United States"] or a in ["Mexico","Canada","United States"])
        f = np.array([[elo_h, elo_a, int(not neutral), elo_h-elo_a]])
        prob_h = model.predict_proba(f)[0,1]
        prob_d = (1-prob_h)*0.45; prob_a = 1-prob_h-prob_d
        xg_h = 1.2 + prob_h*2
        xg_a = 1.2 + prob_a*2
        data[fecha_str].append({
            "local": h, "visitante": a, "grupo": p["grupo"], "hora": p["hora"],
            "prob_h": round(prob_h*100,1), "prob_d": round(prob_d*100,1), "prob_a": round(prob_a*100,1),
            "xg_h": round(xg_h,2), "xg_a": round(xg_a,2),
            "over25": round(xg_h+xg_a > 2.5,1), "ambos": round(xg_h>0.8 and xg_a>0.8,1),
            "marcador": f"{round(xg_h)}-{round(xg_a)}"
        })

with open("data.json", "w") as f:
    json.dump(data, f, ensure_ascii=False)
print("✅ Predicciones generadas y guardadas en data.json.")
