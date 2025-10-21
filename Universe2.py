import os, json, datetime
import numpy as np, pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPRegressor
import time

time.sleep(30)

SEED = 42
np.random.seed(SEED)

outdir = "/mnt/data/universal_synthetic"
os.makedirs(outdir, exist_ok=True)

# Smaller dataset for faster execution
N = 1500
constellations = ["Orion","Lyra","Cygnus","Draco","Andromeda","Pegasus","Scorpius","Cassiopeia","Phoenix","Vulpecula"]
ra = np.random.uniform(0,360,size=N)
dec = np.random.uniform(-90,90,size=N)
distance_ly = np.random.exponential(scale=800,size=N) + np.random.normal(0,40,size=N)
distance_ly = np.clip(distance_ly,0.1,None)
redshift = np.abs(np.random.normal(0.02,0.01,size=N)) + (distance_ly / 1e6)
luminosity = 10 ** (np.random.normal(28,1.2,size=N))
dark_matter_density = np.abs(np.random.normal(0.3,0.1,size=N)) * (1 + (distance_ly/1e5)*0.02)
energy_flux = luminosity / (4 * np.pi * (distance_ly*9.461e15)**2 + 1e-12)
planet_count = np.random.poisson(2,size=N)
avg_planet_mass = np.random.lognormal(mean=np.log(1.0), sigma=1.0, size=N) * (1 + (planet_count>0)*0.2)
orbital_speed_mean = np.abs(np.random.normal(30,20,size=N)) + (avg_planet_mass**0.25)
star_temperature = np.random.normal(5000,1200,size=N)
metallicity = np.random.normal(0.02,0.01,size=N)
gravitational_well = np.log1p(luminosity) * (dark_matter_density**0.5) / (distance_ly**0.1 + 1e-6)
measurement_noise = np.random.normal(0,0.05,size=N)
const_idx = np.random.choice(len(constellations), size=N)
constellation_name = [constellations[i] for i in const_idx]
quantum_fluctuation_index = np.abs(np.random.normal(0.001,0.0008,size=N)) * (1 + np.sin(ra/180*np.pi))
dark_energy_pressure = np.random.uniform(-1.2,-0.6,size=N) + (redshift*0.03)

df = pd.DataFrame({
    "obs_id": np.arange(1,N+1),
    "ra_deg": ra,
    "dec_deg": dec,
    "distance_ly": distance_ly,
    "redshift": redshift,
    "luminosity": luminosity,
    "dark_matter_density": dark_matter_density,
    "energy_flux": energy_flux,
    "planet_count": planet_count,
    "avg_planet_mass": avg_planet_mass,
    "orbital_speed_mean_km_s": orbital_speed_mean,
    "star_temperature_K": star_temperature,
    "metallicity": metallicity,
    "gravitational_well": gravitational_well,
    "measurement_noise": measurement_noise,
    "quantum_fluctuation_index": quantum_fluctuation_index,
    "dark_energy_pressure": dark_energy_pressure,
    "constellation": constellation_name
})

df["anomaly_score"] = (df["dark_matter_density"] * 2.0 + df["quantum_fluctuation_index"]*1000) / (1 + df["distance_ly"]/1e4)
df["is_anomalous"] = (df["anomaly_score"] > np.percentile(df["anomaly_score"], 95)).astype(int)
csv_path = os.path.join(outdir, "synthetic_universal_data_small.csv")
df.to_csv(csv_path, index=False)

# Preprocess
feature_cols = [
    "ra_deg","dec_deg","distance_ly","redshift","luminosity","dark_matter_density",
    "energy_flux","planet_count","avg_planet_mass","orbital_speed_mean_km_s",
    "star_temperature_K","metallicity","gravitational_well",
    "quantum_fluctuation_index","dark_energy_pressure"
]
X = df[feature_cols].copy()
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_train, X_val = train_test_split(X_scaled, test_size=0.2, random_state=SEED)

# Lighter MLP autoencoder
hidden_layers = (48, 24, 8, 3, 8, 24, 48)
mlp = MLPRegressor(hidden_layer_sizes=hidden_layers, activation='relu', solver='adam',
                   max_iter=60, random_state=SEED, verbose=True)
mlp.fit(X_train, X_train)

# Function to get latent activations
def mlp_forward_get_latent(mlp_model, X_input, latent_layer_index=3):
    activ = X_input
    coefs = mlp_model.coefs_
    intercepts = mlp_model.intercepts_
    for i, (W, b) in enumerate(zip(coefs, intercepts)):
        activ = activ.dot(W) + b
        if i < len(coefs)-1:
            activ = np.where(activ > 0, activ, 0.0)
        if i == latent_layer_index:
            return activ
    return activ

latent_vectors = mlp_forward_get_latent(mlp, X_scaled, latent_layer_index=3)
latent_df = pd.DataFrame(latent_vectors, columns=[f"z{i+1}" for i in range(latent_vectors.shape[1])])
latent_df["constellation"] = df["constellation"].values
latent_df["is_anomalous"] = df["is_anomalous"].values
latent_csv = os.path.join(outdir, "latent_vectors_small.csv")
latent_df.to_csv(latent_csv, index=False)

# Quick visualizations (kept minimal to save time): PCA of latent, 3D latent scatter, histograms
pca = PCA(n_components=2, random_state=SEED)
pca_2d = pca.fit_transform(latent_vectors)
plt.figure(figsize=(6,5))
plt.scatter(pca_2d[:,0], pca_2d[:,1], s=8, alpha=0.7)
plt.title("PCA of Learned Latent Space (2D)")
plt.xlabel("PC1")
plt.ylabel("PC2")
pca_path = os.path.join(outdir, "latent_pca_small.png")
plt.savefig(pca_path)
plt.show()

from mpl_toolkits.mplot3d import Axes3D  # noqa
fig = plt.figure(figsize=(7,6))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(latent_vectors[:,0], latent_vectors[:,1], latent_vectors[:,2], s=6, alpha=0.6)
ax.set_title("3D Learned Latent Space (small)")
ax.set_xlabel("z1"); ax.set_ylabel("z2"); ax.set_zlabel("z3")
latent_3d_path = os.path.join(outdir, "latent_3d_small.png")
plt.savefig(latent_3d_path)
plt.show()

# Histograms (a couple)
for feat in ["dark_matter_density","distance_ly","luminosity"]:
    plt.figure(figsize=(6,4))
    plt.hist(df[feat].values, bins=50)
    plt.title(f"Histogram: {feat}")
    plt.xlabel(feat); plt.ylabel("Count")
    fname = os.path.join(outdir, f"hist_{feat}_small.png")
    plt.savefig(fname)
    plt.show()

# Save artifacts summary
artifact_list = {
    "csv_dataset": csv_path,
    "latent_csv": latent_csv,
    "latent_pca": pca_path,
    "latent_3d": latent_3d_path,
    "mlp_saved": None,
    "generated_at": datetime.datetime.utcnow().isoformat() + "Z",
    "num_observations": int(N),
    "features": feature_cols
}
try:
    import joblib
    mlp_path = os.path.join(outdir, "mlp_autoencoder_small.joblib")
    joblib.dump(mlp, mlp_path)
    artifact_list["mlp_saved"] = mlp_path
except Exception as e:
    artifact_list["mlp_saved"] = "joblib not available: " + str(e)

meta_path = os.path.join(outdir, "artifacts_summary_small.json")
with open(meta_path, "w") as f:
    json.dump(artifact_list, f, indent=2)

print("\nArtifacts saved to:", outdir)
for k,v in artifact_list.items():
    print(" -", k, ":", v)

print("\nTop anomalous sample (preview):")
display(df.sort_values("anomaly_score", ascending=False).head(8)[["obs_id","constellation","distance_ly","dark_matter_density","anomaly_score","is_anomalous"]])

print("\nDownload folder: '/mnt/data/universal_synthetic'")
