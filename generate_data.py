import pandas as pd, numpy as np, math, json, os, urllib.request, ssl
from collections import defaultdict
from datetime import date, timedelta

HOY = date.today()
ssl._create_default_https_context = ssl._create_unverified_context

DATA_URL = "https://raw.githubusercontent.com/martj42/international_results/master/results.csv"
if not os.path.exists("internationals.csv"):
    urllib.request.urlretrieve(DATA_URL, "internationals.csv")

df = pd.read_csv("internationals.csv", usecols=['date','home_team','away_team','home_score','away_score'])
df = df.dropna(subset=['home_score','away_score'])
df['date'] = pd.to_datetime(df['date'], errors='coerce')
df = df.dropna(subset=['date']).sort_values('date').tail(2000).copy()

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

# Calendario completo
grupos = {
    "A": ["Mexico","Canada","France","Peru"], "B": ["Brazil","Serbia","England","Iran"],
    "C": ["Argentina","Egypt","Spain","Australia"], "D": ["Germany","Morocco","Netherlands","Japan"],
    "E": ["Uruguay","South Korea","Belgium","Panama"], "F": ["Portugal","Ghana","United States","Switzerland"],
    "G": ["Italy","Nigeria","Colombia","Saudi Arabia"], "H": ["Senegal","Denmark","Chile","Qatar"],
    "I": ["Croatia","Russia","Cameroon","Jamaica"], "J": ["Spain","Scotland","Norway","Costa Rica"],
    "K": ["Belgium","Morocco","Canada","Panama"], "L": ["Portugal","Uruguay","South Korea","Ghana"],
}

enfrentamientos = [(0,1),(2,3),(0,2),(1,3),(0,3),(1,2)]
calendario = {}
inicio = date(2026,6,11)
for g, eqs in grupos.items():
    for i,(e1,e2) in enumerate(enfrentamientos):
        dia = inicio + timedelta(days=i//4)
        key = dia.isoformat()
        if key not in calendario: calendario[key] = []
        h,a = eqs[e1], eqs[e2]
        elo_h, elo_a = elo.get(h,1500), elo.get(a,1500)
        neutral = not (h in ["Mexico","Canada","United States"] or a in ["Mexico","Canada","United States"])
        f = np.array([[elo_h, elo_a, int(not neutral), elo_h-elo_a]])
        prob_h = model.predict_proba(f)[0,1]
        prob_d = (1-prob_h)*0.45; prob_a = 1-prob_h-prob_d
        xg_h = 1.2 + prob_h*2; xg_a = 1.2 + prob_a*2
        calendario[key].append({
            "local": h, "visitante": a, "grupo": g, "hora": f"{12+(i%4)}:00",
            "fase": "Grupos",
            "prob_h": round(prob_h*100,1), "prob_d": round(prob_d*100,1), "prob_a": round(prob_a*100,1),
            "xg_h": round(xg_h,2), "xg_a": round(xg_a,2),
            "over25": xg_h+xg_a > 2.5, "ambos": xg_h>0.8 and xg_a>0.8,
            "marcador": f"{round(xg_h)}-{round(xg_a)}",
            "cuota_h": round(1/prob_h*0.95,2) if prob_h>0 else 999,
            "cuota_d": round(1/prob_d*0.95,2) if prob_d>0 else 999,
            "cuota_a": round(1/prob_a*0.95,2) if prob_a>0 else 999,
            "ev_h": round(prob_h*(1/prob_h*0.95)-1,3) if prob_h>0 else -1,
            "ev_a": round(prob_a*(1/prob_a*0.95)-1,3) if prob_a>0 else -1
        })

# Añadir eliminatorias (con nombres de cruces, no equipos TBD)
eliminatorias = [
    ("2026-06-28", "R32", "1°A", "2°B"), ("2026-06-28", "R32", "1°B", "2°A"),
    ("2026-06-28", "R32", "1°C", "2°D"), ("2026-06-28", "R32", "1°D", "2°C"),
    ("2026-06-29", "R32", "1°E", "2°F"), ("2026-06-29", "R32", "1°F", "2°E"),
    ("2026-06-29", "R32", "1°G", "2°H"), ("2026-06-29", "R32", "1°H", "2°G"),
    ("2026-06-30", "R32", "1°I", "2°J"), ("2026-06-30", "R32", "1°J", "2°I"),
    ("2026-06-30", "R32", "1°K", "2°L"), ("2026-06-30", "R32", "1°L", "2°K"),
    ("2026-07-01", "R32", "Mejor 3°", "Mejor 3°"), ("2026-07-01", "R32", "Mejor 3°", "Mejor 3°"),
    ("2026-07-01", "R32", "Mejor 3°", "Mejor 3°"), ("2026-07-01", "R32", "Mejor 3°", "Mejor 3°"),
    ("2026-07-04", "Octavos", "Ganador R32", "Ganador R32"),
    ("2026-07-05", "Octavos", "Ganador R32", "Ganador R32"),
    ("2026-07-06", "Octavos", "Ganador R32", "Ganador R32"),
    ("2026-07-07", "Octavos", "Ganador R32", "Ganador R32"),
    ("2026-07-10", "Cuartos", "Ganador Octavos", "Ganador Octavos"),
    ("2026-07-11", "Cuartos", "Ganador Octavos", "Ganador Octavos"),
    ("2026-07-14", "Semifinal", "Ganador Cuartos", "Ganador Cuartos"),
    ("2026-07-15", "Semifinal", "Ganador Cuartos", "Ganador Cuartos"),
    ("2026-07-18", "3er Puesto", "Perdedor Semi", "Perdedor Semi"),
    ("2026-07-19", "FINAL", "Ganador Semi", "Ganador Semi")
]

for fecha, fase, eq1, eq2 in eliminatorias:
    key = fecha
    if key not in calendario: calendario[key] = []
    calendario[key].append({
        "local": eq1, "visitante": eq2, "grupo": fase, "hora": "TBD",
        "fase": fase,
        "prob_h": None, "prob_d": None, "prob_a": None,
        "xg_h": None, "xg_a": None, "over25": None, "ambos": None,
        "marcador": "Por definir",
        "es_tbd": True
    })

with open("data.json", "w") as f:
    json.dump(calendario, f, ensure_ascii=False)
print(f"✅ Calendario generado con {len(calendario)} fechas.")
