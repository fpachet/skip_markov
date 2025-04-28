import mido
import numpy as np
from collections import defaultdict
from scipy.special import softmax
from scipy.optimize import minimize
import random

def extract_notes(path):
    """Extracts MIDI note sequence from a MIDI file."""
    mid = mido.MidiFile(path)
    notes = []
    if len(mid.tracks) == 1:
        track = mid.tracks[0]
    else:
        track = mid.tracks[1]
    for msg in track:
        if msg.type == "note_on" and msg.velocity > 0:
            notes.append(msg.note)
    return notes

def save_midi(sequence, output_file="generated.mid"):
    """Saves a sequence of notes to a MIDI file."""
    mid = mido.MidiFile()
    track = mido.MidiTrack()
    mid.tracks.append(track)
    for note in sequence:
        track.append(mido.Message("note_on", note=note, velocity=64, time=0))
        track.append(mido.Message("note_off", note=note, velocity=64, time=120))
    mid.save(output_file)

def train_skip_markov(train_seq, skip):
    """Train a first-order skip=\u0394 Markov model."""
    transition_counts = defaultdict(lambda: defaultdict(float))
    T = len(train_seq)
    for t in range(T - 1):
        context_idx = t + 1 - skip
        if 0 <= context_idx < T:
            prev_note = train_seq[context_idx]
            next_note = train_seq[t + 1]
            transition_counts[prev_note][next_note] += 1.0

    model = {}
    for prev_note, next_counts in transition_counts.items():
        total = sum(next_counts.values())
        model[prev_note] = {nxt: cnt / total for nxt, cnt in next_counts.items()}
    return model

def train_all_skip_models(train_seq, skip_distances):
    """Train multiple skip=\u0394 models."""
    skip_models = {}
    for d in skip_distances:
        skip_models[d] = train_skip_markov(train_seq, d)
    return skip_models

def build_mixture_matrix(train_seq, skip_models, skip_distances):
    """Build P matrix for mixture fitting."""
    T = len(train_seq)
    D = len(skip_distances)
    P = np.zeros((T - 1, D))
    for j, skip in enumerate(skip_distances):
        model = skip_models[skip]
        for t in range(T - 1):
            context_idx = t + 1 - skip
            if 0 <= context_idx < T:
                prev_note = train_seq[context_idx]
                next_note = train_seq[t + 1]
                if prev_note in model and next_note in model[prev_note]:
                    P[t, j] = model[prev_note][next_note]
                else:
                    P[t, j] = 0.0
            else:
                P[t, j] = 0.0
    return P

def mixture_objective_and_grad(u, P):
    """Objective and gradient for mixture model."""
    exp_u = np.exp(u)
    alpha = exp_u / np.sum(exp_u)
    s = P @ alpha
    eps = 1e-12
    s = np.clip(s, eps, None)
    L = np.sum(np.log(s))
    neg_L = -L

    grad = np.zeros_like(u)
    for m in range(len(u)):
        alpha_m = alpha[m]
        grad_part = alpha_m * (P[:, m] - s) / s
        grad[m] = np.sum(grad_part)

    grad = -grad
    return neg_L, grad

def fit_mixture_weights(P, method='BFGS'):
    """Fit mixture weights via optimization."""
    D = P.shape[1]
    u_init = np.zeros(D)
    result = minimize(
        fun=mixture_objective_and_grad,
        x0=u_init,
        args=(P,),
        jac=True,
        method=method
    )
    u_opt = result.x
    alpha = softmax(u_opt)
    return alpha

def generate_sequence(skip_models, skip_distances, alpha, seed_note, gen_len=50):
    """Generate a new sequence."""
    generated = [seed_note]
    D = len(alpha)
    all_notes = set()
    for m in skip_models.values():
        all_notes.update(m.keys())
        for d in m.values():
            all_notes.update(d.keys())
    all_notes = sorted(all_notes)
    vocab_size = len(all_notes)

    for t in range(gen_len - 1):
        mixture_dist = defaultdict(float)
        for j, d in enumerate(skip_distances):
            idx = (len(generated)) - d
            if 0 <= idx < len(generated):
                prev_note = generated[idx]
                if prev_note in skip_models[d]:
                    local_dist = skip_models[d][prev_note]
                    for candidate_next, p_local in local_dist.items():
                        mixture_dist[candidate_next] += alpha[j] * p_local
        total_prob = sum(mixture_dist.values())
        if total_prob < 1e-12:
            for n in all_notes:
                mixture_dist[n] = 1.0 / vocab_size
            total_prob = 1.0

        for k in mixture_dist:
            mixture_dist[k] /= total_prob

        notes_list = list(mixture_dist.keys())
        probs_list = np.array([mixture_dist[n] for n in notes_list])
        cumsum_probs = np.cumsum(probs_list)
        r = random.random()
        idx_sample = np.searchsorted(cumsum_probs, r)
        chosen_note = notes_list[idx_sample]
        generated.append(chosen_note)

    return generated

def metropolis_resample_sequence(skip_models, skip_distances, alpha, vocab, seq_len=50, num_iters=5000):
    """Generate a new sequence using Metropolis resampling."""
    generated = random.choices(vocab, k=seq_len)
    D = len(alpha)

    for _ in range(num_iters):
        t = random.randint(1, seq_len - 2)  # avoid boundaries

        mixture_dist = defaultdict(float)

        for j, d in enumerate(skip_distances):
            context_idx = t - d
            if 0 <= context_idx < seq_len:
                context_note = generated[context_idx]
                if context_note in skip_models[d]:
                    local_dist = skip_models[d][context_note]
                    for candidate_next, p_local in local_dist.items():
                        mixture_dist[candidate_next] += alpha[j] * p_local

        if not mixture_dist:
            continue

        total_prob = sum(mixture_dist.values())
        for k in mixture_dist:
            mixture_dist[k] /= total_prob

        notes_list = list(mixture_dist.keys())
        probs_list = np.array([mixture_dist[n] for n in notes_list])
        cumsum_probs = np.cumsum(probs_list)
        r = random.random()
        idx_sample = np.searchsorted(cumsum_probs, r)
        chosen_note = notes_list[idx_sample]

        generated[t] = chosen_note

    return generated

def smooth_sequence(sequence, skip_models, skip_distances, alpha):
    """Generate a new sequence using Metropolis resampling."""
    seq_len = len(sequence)
    for t in range(len(sequence)):
        mixture_dist = defaultdict(float)
        for j, d in enumerate(skip_distances):
            context_idx = t - d
            if 0 <= context_idx < seq_len:
                context_note = sequence[context_idx]
                if context_note in skip_models[d]:
                    local_dist = skip_models[d][context_note]
                    for candidate_next, p_local in local_dist.items():
                        mixture_dist[candidate_next] += alpha[j] * p_local

        if not mixture_dist:
            continue
        total_prob = sum(mixture_dist.values())
        for k in mixture_dist:
            mixture_dist[k] /= total_prob
        notes_list = list(mixture_dist.keys())
        probs_list = np.array([mixture_dist[n] for n in notes_list])
        np.argmax(probs_list)
        idx_sample = np.argmax(probs_list)
        chosen_note = notes_list[idx_sample]
        sequence[t] = chosen_note
    return sequence

def main_example():
    path = "../data/bach_partita_violin.mid"
    train_seq = extract_notes(path)

    # skip_distances = [-4, -3, -2, -1, 1, 2, 3, 4]
    skip_distances = [-3, -2, -1, 1, 2, 3]

    skip_models = train_all_skip_models(train_seq, skip_distances)
    P = build_mixture_matrix(train_seq, skip_models, skip_distances)
    alpha = fit_mixture_weights(P, method='BFGS')
    print("Learned mixture weights:", alpha)

    # seed_note = train_seq[0]
    # gen_seq = generate_sequence(skip_models, skip_distances, alpha, seed_note, gen_len=200)
    all_notes = sorted(set(train_seq))
    gen_seq = metropolis_resample_sequence(skip_models, skip_distances, alpha, vocab=all_notes, seq_len=2000, num_iters=10000)
    gen_seq = smooth_sequence(gen_seq, skip_models, skip_distances, alpha)
    save_midi(gen_seq, output_file="../skip_futures_generated.mid")

if __name__ == "__main__":
    main_example()
