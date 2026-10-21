import numpy as np
from scipy.signal import butter, filtfilt
from scipy.stats import linregress
from scipy.spatial import KDTree

# ------------------ FILTER ------------------
def butter_filter(x, fs=300):
    """
    Filtre passe-haut du signal ECG.
    - Fréquence de coupure : 0.67 Hz (pour enlever la ligne de base)
    - Filtre Butterworth d'ordre 3, phase nulle (filtfilt)
    - Le signal est divisé par 1000 (conversion µV -> mV si nécessaire)
    """
    b, a = butter(3, 0.67 / (fs / 2), btype="high")
    return filtfilt(b, a, x / 1000.0)

# ------------------ DFA (Detrended Fluctuation Analysis) ------------------
def DFA(DATA, win_length, order=1):
    """
    Calcule la fluctuation pour une échelle donnée (fenêtre de longueur win_length).
    """
    DATA = DATA.flatten()
    N = len(DATA)
    n = N // win_length
    if n == 0:
        return np.nan
    DATA = DATA[:n * win_length]
    y = np.cumsum(DATA - np.mean(DATA))
    Yn = np.zeros_like(y)
    x = np.arange(1, win_length + 1)
    for j in range(n):
        idx = slice(j * win_length, (j + 1) * win_length)
        # Régression polynomiale d'ordre 'order' sur la fenêtre
        coeffs = np.polyfit(x, y[idx], order)
        Yn[idx] = np.polyval(coeffs, x)
    F = np.sqrt(np.mean((y - Yn) ** 2))
    return F

def DFA_main(DATA):
    """
    Calcule l'exposant α₁ (pente) du DFA pour des échelles de 100 à 1000 par pas de 100.
    Retourne (3 - α₁, α₁) – le premier élément est ignoré dans l'appel.
    """
    n_values = np.arange(100, 1100, 100)
    valid_n = n_values[n_values <= len(DATA)]
    if len(valid_n) < 2:
        slope = 1.0
        return 3 - slope, slope
    F_n = np.array([DFA(DATA, n, 1) for n in valid_n])
    mask = np.isfinite(F_n) & (F_n > 0)
    if np.sum(mask) < 2:
        slope = 1.0
        return 3 - slope, slope
    slope, *_ = linregress(np.log(valid_n[mask]), np.log(F_n[mask]))
    return 3 - slope, slope  # (intercept ignoré, α₁)

# ------------------ Lyapunov Exponent ------------------
def lyapunovExponent(x, fs, m=10, tau=1, max_iter=200):
    """
    Estime le plus grand exposant de Lyapunov à partir d'une série temporelle.
    - Reconstruction de l'espace des phases par la méthode des délais (dimension m, délai tau)
    - Recherche du plus proche voisin pour chaque point (excluant le point lui-même)
    - Suivi de l'évolution de la distance euclidienne entre les trajectoires
    - Régression linéaire sur le logarithme de la divergence moyenne
    """
    x = np.asarray(x).flatten()
    N = len(x) - (m - 1) * tau
    if N <= 1:
        return 0.0

    # Construction de l'espace des phases : matrice (N, m)
    Y = np.column_stack([x[i * tau : i * tau + N] for i in range(m)])
    if Y.shape[0] < 2:
        return 0.0

    # Arbre KD pour trouver les plus proches voisins
    tree = KDTree(Y)
    distances, indices = tree.query(Y, k=2)  # k=2 : le premier est le point lui-même
    nn_dist = distances[:, 1]                # distance au plus proche voisin
    nn_idx = indices[:, 1]                   # indice du plus proche voisin

    L = min(max_iter, N)
    if L <= 1:
        return 0.0

    # Suivi de la divergence, vectorisé par pas de temps k.
    # Pour un k donné, les points futurs sont Y[i+k] et Y[nn_idx[i]+k] pour tous
    # les i tels que i+k < N et nn_idx[i]+k < N. On les évalue tous d'un coup
    # (au lieu d'une double boucle Python + np.linalg.norm appelée 600 000 fois).
    div = np.zeros(L)
    count = np.zeros(L)
    base_idx = np.arange(N)
    for k in range(L):
        valid = (base_idx + k < N) & (nn_idx + k < N)
        n_valid = int(valid.sum())
        if n_valid == 0:
            continue
        i_idx = base_idx[valid] + k
        j_idx = nn_idx[valid] + k
        d = np.sqrt(np.sum((Y[i_idx] - Y[j_idx]) ** 2, axis=1))
        div[k] = d.sum()
        count[k] = n_valid

    # Moyenne de la divergence pour chaque pas de temps
    div = div[count > 0] / count[count > 0]
    if div.size < 2:
        return 0.0

    # Régression sur les points où la divergence est positive (log)
    pos = div > 0
    if np.sum(pos) < 2:
        return 0.0

    t = np.arange(len(div))[pos] / fs
    slope, *_ = linregress(t, np.log(div[pos]))
    return slope
