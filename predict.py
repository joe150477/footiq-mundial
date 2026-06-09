import pandas as pd, numpy as np, math, os, json, datetime, urllib.request, ssl, joblib
from collections import defaultdict
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression

HOY = datetime.date.today()
DATA_DIR = "data"
MODEL_DIR = "models"
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

# Descargar dataset si no existe
csv_path = os.path.join(DATA_DIR, "internationals.csv")
if not os.path.exists(csv_path):
    ssl._create_default_https_context = ssl._create_unverified_context
    urllib.request.urlretrieve("https://raw.githubusercontent.com/martj42/international_results/master/results.csv", csv_path)

df = pd.read_csv(csv_path, usecols=['date','home_team','away_team','home_score','away_score'])
df = df.dropna(subset=['home_score','away_score'])
df['date'] = pd.to_datetime(df['date'], errors='coerce')
df = df.dropna(subset=['date']).sort_values('date').tail(2000).copy()

# Entrenar modelo rápido (igual que en FootIQ)
K=20; elo=defaultdict(lambda:1500); X,y=[],[]
for _,row in df.iterrows():
    h,a,gh,ga = row['home_team'],row['away_team'],row['home_score'],row['away_score']
    Rh,Ra=elo[h],elo[a]; avg_elo=sum(elo.values())/len(elo) if elo else 1500
    neutral = not (h in ["Mexico","Canada","United States"] or a in ["Mexico","Canada","United States"])
    base_h,base_a=(1.2,1.2) if neutral else (1.4,1.1)
    fh=math.exp((Rh-avg_elo)/400); fa=math.exp((Ra-avg_elo)/400)
    lam_h=max(0.3,min(4.0,base_h*fh/(fa**0.3))); lam_a=max(0.3,min(4.0,base_a*fa/(fh**0.3)))
    win=0
    for i in range(9):
        for j in range(9):
            p = (lam_h**i*math.exp(-lam_h)/math.factorial(i))*(lam_a**j*math.exp(-lam_a)/math.factorial(j))
            if i>j: win+=p
    X.append([Rh,Ra,int(not neutral),Rh-Ra])
    y.append(1 if gh>ga else 0)
    Eh=1/(1+10**((Ra-Rh)/400))
    if gh>ga: Sh,Sa=1,0
    elif gh==ga: Sh,Sa=0.5,0.5
    else: Sh,Sa=0,1
    elo[h]=Rh+K*(Sh-Eh); elo[a]=Ra+K*(Sa-(1-Eh))

model = LogisticRegression(solver='lbfgs')
model.fit(X,y)
cal = IsotonicRegression(out_of_bounds='clip')
cal.fit(model.predict_proba(X)[:,1], y)

# Calendario de grupos
grupos = {
    "A": ["Mexico","Canada","France","Peru"], "B": ["Brazil","Serbia","England","Iran"],
    "C": ["Argentina","Egypt","Spain","Australia"], "D": ["Germany","Morocco","Netherlands","Japan"],
    "E": ["Uruguay","South Korea","Belgium","Panama"], "F": ["Portugal","Ghana","United States","Switzerland"],
    "G": ["Italy","Nigeria","Colombia","Saudi Arabia"], "H": ["Senegal","Denmark","Chile","Qatar"],
    "I": ["Croatia","Russia","Cameroon","Jamaica"], "J": ["Spain","Scotland","Norway","Costa Rica"],
    "K": ["Belgium","Morocco","Canada","Panama"], "L": ["Portugal","Uruguay","South Korea","Ghana"],
}
partidos_hoy = []
for g, eqs in grupos.items():
    for i,j in [(0,1),(2,3),(0,2),(1,3),(0,3),(1,2)]:
        partidos_hoy.append((g, eqs[i], eqs[j]))

# Para la demo, solo predecimos algunos (en realidad filtraríamos por fecha)
predicciones = {"fecha": str(HOY), "partidos": []}
for grupo, h, a in partidos_hoy[:4]:  # limitamos a 4 para no saturar
    elo_h = elo.get(h,1500); elo_a = elo.get(a,1500)
    neutral = not (h in ["Mexico","Canada","United States"] or a in ["Mexico","Canada","United States"])
    features = np.array([[elo_h, elo_a, int(not neutral), elo_h-elo_a]])
    prob_h = cal.predict([model.predict_proba(features)[0,1]])[0]
    prob_d = (1-prob_h)*0.45
    prob_a = (1-prob_h)-prob_d
    predicciones["partidos"].append({
        "local": h, "visitante": a,
        "prob_local": round(prob_h*100,1), "prob_empate": round(prob_d*100,1), "prob_visitante": round(prob_a*100,1),
        "marcador": f"{round(prob_h*3)}-{round(prob_a*2)}"
    })

with open("data.json", "w") as f:
    json.dump(predicciones, f)

# Actualizar index.html con los nuevos datos
html = f'''<!DOCTYPE html>
<html lang="es">
<head>
    <meta charset="UTF-8"><meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>FootIQ Mundial 2026</title>
    <link rel="manifest" href="data:application/json;base64,eyJuYW1lIjoiRm9vdElRIiwic2hvcnRfbmFtZSI6IkZvb3RJUSIsInN0YXJ0X3VybCI6Ii4iLCJkaXNwbGF5Ijoic3RhbmRhbG9uZSJ9">
    <style>body{{font-family:sans-serif;background:#0f172a;color:#e2e8f0;padding:16px}}h1{{color:#38bdf8}}.card{{background:#1e293b;padding:12px;margin:10px 0;border-radius:12px}}.prob{{color:#4ade80;font-weight:bold}}</style>
</head>
<body>
    <h1>🌍 FootIQ · {HOY.strftime('%d/%m/%Y')}</h1>
    <div id="app"></div>
    <script>
        const DATA = {json.dumps(predicciones, ensure_ascii=False)};
        let html = '';
        DATA.partidos.forEach(p => {{
            html += `<div class="card">
                <p><strong>${{p.local}}</strong> <span class="prob">${{p.prob_local}}%</span></p>
                <p>Empate ${{p.prob_empate}}%</p>
                <p><strong>${{p.visitante}}</strong> <span class="prob">${{p.prob_visitante}}%</span></p>
                <p>🎯 Marcador probable: ${{p.marcador}}</p>
            </div>`;
        }});
        if (!DATA.partidos.length) html = '<p>No hay partidos hoy.</p>';
        document.getElementById('app').innerHTML = html;
    </script>
</body>
</html>'''

with open("index.html", "w") as f:
    f.write(html)
print("Predicciones actualizadas.")
