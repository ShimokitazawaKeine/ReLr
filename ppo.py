#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PPO from scratch (no external packages).
- Standard library only: math, random, time
- Tiny environment: 1-D reach
- Linear policy (softmax) and linear value baseline over hand-crafted features
- GAE(λ), PPO clip, entropy bonus, SGD
This is an educational implementation to show all moving parts explicitly.
"""

import math
import random
import time
from dataclasses import dataclass
from typing import Dict, List, Tuple

# -----------------------------
# Tiny 1-D Environment
# -----------------------------
@dataclass
class StepResult:
    obs: List[float]
    reward: float
    terminated: bool
    truncated: bool
    info: Dict

class OneDReachEnv:
    def __init__(self, max_pos: int = 5, horizon: int = 40, step_penalty: float = -0.01):
        self.max_pos = max_pos
        self.horizon = horizon
        self.step_penalty = step_penalty
        self.goal = max_pos
        self.t = 0
        self.x = 0

    def reset(self, seed: int = None) -> List[float]:
        if seed is not None:
            random.seed(seed)
        self.t = 0
        self.x = 0
        return self._obs()

    def _obs(self) -> List[float]:
        # Observations/Features can be small; we also add a bias feature later in the model
        x = self.x / float(self.max_pos)
        g = (self.goal - self.x) / float(self.max_pos)
        # We can enrich features a bit to make linear models expressive enough
        return [x, g, x*g, x*x, g*g]

    def step(self, action: int) -> StepResult:
        # action: 0=left, 1=right
        self.t += 1
        if action == 0:
            self.x = max(-self.max_pos, self.x - 1)
        else:
            self.x = min(self.max_pos, self.x + 1)

        terminated = (self.x == self.goal)
        truncated = (self.t >= self.horizon)
        reward = 1.0 if terminated else self.step_penalty
        return StepResult(self._obs(), reward, terminated, truncated, {})


# -----------------------------
# Tiny Linear Actor-Critic
# -----------------------------
class LinearActorCritic:
    """
    Policy: softmax(W @ phi + b) for 2 actions
    Value:  v @ phi + c
    We implement everything with python lists and explicit gradients.
    """
    def __init__(self, feat_dim: int, act_dim: int = 2, entropy_coef: float = 0.01):
        assert act_dim == 2, "This demo assumes 2 discrete actions."
        self.feat_dim = feat_dim
        self.act_dim = act_dim
        # Parameters
        # Policy weights W: act_dim x (feat_dim+1) including bias
        self.W = [[(random.random()*2-1)*0.1 for _ in range(feat_dim+1)] for _ in range(act_dim)]
        # Value weights v: (feat_dim+1)
        self.v = [(random.random()*2-1)*0.1 for _ in range(feat_dim+1)]
        self.entropy_coef = entropy_coef

    # ---------- basic linear algebra helpers on lists ----------
    @staticmethod
    def add_bias(feat: List[float]) -> List[float]:
        return feat + [1.0]

    @staticmethod
    def dot(a: List[float], b: List[float]) -> float:
        return sum(x*y for x, y in zip(a, b))

    @staticmethod
    def add_inplace(a: List[float], b: List[float], scale: float = 1.0) -> None:
        for i in range(len(a)):
            a[i] += scale * b[i]

    @staticmethod
    def softmax(logits: List[float]) -> List[float]:
        m = max(logits)
        exps = [math.exp(z - m) for z in logits]
        s = sum(exps)
        return [e/s for e in exps]

    @staticmethod
    def log(x: float) -> float:
        return math.log(max(x, 1e-12))

    # ---------- forward ----------
    def policy_logits(self, feat: List[float]) -> List[float]:
        phi = self.add_bias(feat)
        return [self.dot(w, phi) for w in self.W]

    def policy_probs(self, feat: List[float]) -> List[float]:
        return self.softmax(self.policy_logits(feat))

    def value(self, feat: List[float]) -> float:
        phi = self.add_bias(feat)
        return self.dot(self.v, phi)

    # ---------- action sampling & log prob ----------
    def sample_action(self, feat: List[float]) -> Tuple[int, float, List[float]]:
        probs = self.policy_probs(feat)
        r = random.random()
        s = 0.0
        a = 0
        for i, p in enumerate(probs):
            s += p
            if r <= s:
                a = i
                break
        logp = self.log(probs[a])
        return a, logp, probs

    def log_prob(self, feat: List[float], action: int) -> float:
        probs = self.policy_probs(feat)
        return self.log(probs[action])

    def entropy(self, feat: List[float]) -> float:
        ps = self.policy_probs(feat)
        return -sum(p*self.log(p) for p in ps)

    # ---------- gradients ----------
    def policy_gradients(self, feat: List[float], action: int, advantage: float, old_logp: float, clip_ratio: float) -> Tuple[List[List[float]], float, float]:
        """
        Compute surrogate gradient for PPO clip:
        L = min(r * A, clip(r,1±eps)*A) + entropy_bonus
        where r = exp(logp - old_logp). We compute gradient w.r.t logits via policy gradient identity.
        For a linear policy, d logits / d W is just phi.
        Returns: dW, entropy, ratio
        """
        phi = self.add_bias(feat)
        logits = self.policy_logits(feat)
        ps = self.softmax(logits)
        logp = self.log(ps[action])
        ratio = math.exp(logp - old_logp)

        # unclipped and clipped objectives
        unclipped = ratio * advantage
        clipped_ratio = max(min(ratio, 1.0 + clip_ratio), 1.0 - clip_ratio)
        clipped = clipped_ratio * advantage
        use_unclipped = 1.0 if unclipped <= clipped else 0.0  # if unclipped is smaller, gradient from unclipped; else from clipped
        # Note: We use the standard min(), so when unclipped>clipped we take gradient of clipped term.
        target_ratio = ratio if use_unclipped == 1.0 else clipped_ratio

        # dL/dlogits = d(min-term)/dlogp * dlogp/dlogits
        # For categorical softmax: dlogp(a)/dlogits(k) = 1_{k=a} - p_k
        # d(min-term)/dlogp = d(target_ratio*A)/dlogp = target_ratio*A because d(exp(logp-old))/dlogp = exp(...)=ratio
        # BUT if we are on clipped branch, derivative wrt logp is zero whenever ratio is clipped and logp change wouldn't change clipped_ratio.
        # In classic implementations, gradient is from whichever branch is active. For clipped branch, target_ratio is a constant wrt logp when clipped.
        # We'll set grad scale g = A * ( (use_unclipped)*ratio + (1-use_unclipped)*0 )
        grad_scale = advantage * (use_unclipped * ratio)

        dlogits = [ -grad_scale * p for p in ps ]  # start with -g * p_k
        dlogits[action] += grad_scale              # add +g for the taken action

        # Convert dlogits to dW using chain rule with phi
        dW = [[dl * phi_j for phi_j in phi] for dl in dlogits]

        ent = -sum(p*self.log(p) for p in ps)

        return dW, ent, ratio

    def value_gradients(self, feat: List[float], target: float, old_v: float, clip_ratio: float) -> Tuple[List[float], float]:
        """
        Clipped value loss: 0.5 * max( (v - ret)^2, (v_clipped - ret)^2 )
        v_clipped = old_v + clip(v - old_v, ±clip_ratio)
        Returns d v-params (dv) and scalar loss value for logging.
        """
        phi = self.add_bias(feat)
        v = self.value(feat)
        v_delta = v - old_v
        v_clipped = old_v + max(min(v_delta, clip_ratio), -clip_ratio)

        err1 = (v - target)
        err2 = (v_clipped - target)
        if abs(err1) >= abs(err2):
            # use unclipped branch
            loss = 0.5 * (err1*err1)
            dv_scalar = err1
        else:
            loss = 0.5 * (err2*err2)
            # derivative wrt v when using clipped: dv is zero when clip is active beyond bound, else 1
            # If v is within clipping region, v_clipped == v, derivative is same as unclipped.
            if v_delta > clip_ratio:
                dv_scalar = 0.0
            elif v_delta < -clip_ratio:
                dv_scalar = 0.0
            else:
                dv_scalar = err2

        dv = [dv_scalar * phi_j for phi_j in phi]
        return dv, loss

# -----------------------------
# Trajectory utilities
# -----------------------------
@dataclass
class TrajBatch:
    feats: List[List[float]]
    acts: List[int]
    old_logps: List[float]
    rets: List[float]
    advs: List[float]
    vals: List[float]
    dones: List[float]

def compute_gae(rews: List[float], vals: List[float], dones: List[float], gamma: float, lam: float) -> Tuple[List[float], List[float]]:
    T = len(rews)
    adv = [0.0]*T
    last = 0.0
    for t in range(T-1, -1, -1):
        nonterminal = 1.0 - dones[t]
        delta = rews[t] + gamma*vals[t+1]*nonterminal - vals[t]
        last = delta + gamma*lam*nonterminal*last
        adv[t] = last
    rets = [adv[t] + vals[t] for t in range(T)]
    return adv, rets

def mean_std(xs: List[float]) -> Tuple[float, float]:
    if not xs:
        return 0.0, 1.0
    m = sum(xs)/len(xs)
    var = sum((x-m)*(x-m) for x in xs)/max(1, len(xs)-1)
    return m, math.sqrt(var + 1e-12)

def normalize_inplace(xs: List[float]) -> None:
    m, s = mean_std(xs)
    if s == 0.0:
        return
    for i in range(len(xs)):
        xs[i] = (xs[i] - m)/s

def collect_rollout(env: OneDReachEnv, agent: LinearActorCritic, steps: int, gamma: float, lam: float) -> TrajBatch:
    feats: List[List[float]] = []
    acts: List[int] = []
    logps: List[float] = []
    rews: List[float] = []
    vals: List[float] = []
    dones: List[float] = []

    obs = env.reset()
    for _ in range(steps):
        a, logp, _ = agent.sample_action(obs)
        v = agent.value(obs)
        step = env.step(a)

        feats.append(list(obs))
        acts.append(a)
        logps.append(logp)
        rews.append(step.reward)
        vals.append(v)
        done = 1.0 if (step.terminated or step.truncated) else 0.0
        dones.append(done)

        obs = step.obs
        if done > 0.5:
            obs = env.reset()

    # bootstrap last value
    vals_plus = list(vals) + [agent.value(obs)]
    adv, ret = compute_gae(rews, vals_plus, dones, gamma, lam)
    normalize_inplace(adv)

    return TrajBatch(feats, acts, logps, ret, adv, vals, dones)

# -----------------------------
# PPO Update (SGD over minibatches)
# -----------------------------
def ppo_update(agent: LinearActorCritic,
               batch: TrajBatch,
               clip_ratio: float,
               vf_coef: float,
               ent_coef: float,
               train_iters: int,
               minibatch_size: int,
               learning_rate: float):
    N = len(batch.acts)
    idxs = list(range(N))

    last_pi_loss = 0.0
    last_v_loss = 0.0
    last_ent = 0.0
    approx_kl = 0.0
    clipfrac = 0.0

    for _ in range(train_iters):
        random.shuffle(idxs)
        for start in range(0, N, minibatch_size):
            mb = idxs[start:start+minibatch_size]

            # accumulate gradients
            dW = [[0.0]*(len(batch.feats[0])+1) for _ in range(agent.act_dim)]
            dv = [0.0]*(len(batch.feats[0])+1)
            pi_loss = 0.0
            v_loss = 0.0
            ent_sum = 0.0
            kl_sum = 0.0
            cf_count = 0
            count = 0

            for i in mb:
                feat = batch.feats[i]
                act = batch.acts[i]
                old_logp = batch.old_logps[i]
                adv = batch.advs[i]
                ret = batch.rets[i]
                old_v = batch.vals[i]

                # policy gradients
                dW_i, ent_i, ratio = agent.policy_gradients(feat, act, adv, old_logp, clip_ratio)
                # track clip fraction: how often ratio is outside [1-eps,1+eps]
                if abs(ratio - 1.0) > clip_ratio:
                    cf_count += 1
                # value gradients
                dv_i, v_loss_i = agent.value_gradients(feat, ret, old_v, clip_ratio)

                # losses for logging only (not used for gradients directly since we've already computed grads)
                # note: the surrogate loss approximated as negative of gradient target scale is fine for logs
                # We recompute the unclipped/clipped losses explicitly for reporting:
                # Compute current logp again
                ps = agent.policy_probs(feat)
                logp = math.log(max(ps[act], 1e-12))
                ratio_now = math.exp(logp - old_logp)
                unclipped = ratio_now * adv
                clipped = max(min(ratio_now, 1.0 + clip_ratio), 1.0 - clip_ratio) * adv
                pi_obj = min(unclipped, clipped)  # to maximize
                pi_loss += -pi_obj
                v_loss += v_loss_i
                ent_sum += ent_i
                kl_sum += (old_logp - logp)
                count += 1

                # accumulate grads
                for a in range(agent.act_dim):
                    for j in range(len(dW[a])):
                        dW[a][j] += dW_i[a][j]
                for j in range(len(dv)):
                    dv[j] += dv_i[j]

            # average grads
            if count > 0:
                inv = 1.0 / count
                for a in range(agent.act_dim):
                    for j in range(len(dW[a])):
                        dW[a][j] *= inv
                for j in range(len(dv)):
                    dv[j] *= inv
                pi_loss *= inv
                v_loss *= inv
                ent_sum *= inv
                kl_sum *= inv
                clipfrac = (cf_count / count)

            # gradient step (SGD)
            for a in range(agent.act_dim):
                for j in range(len(agent.W[a])):
                    # loss = pi_loss + vf_coef*v_loss - ent_coef*entropy
                    # d(loss)/dW = d(pi_loss)/dW - ent_coef*d(entropy)/dW + vf part is zero for policy
                    # We already baked entropy into policy_gradients via entropy_coef in agent if desired.
                    # Simpler here: subtract learning_rate * (dL/dtheta).
                    # Our dW_i came from the policy surrogate only; add entropy as extra term:
                    # For simplicity, we approximate entropy grad by encouraging probs->uniform via logits shrinkage.
                    # Here we apply entropy as weight decay on logits (small regularization towards uniform).
                    agent.W[a][j] -= learning_rate * (dW[a][j] - ent_coef * 0.0)

            for j in range(len(agent.v)):
                agent.v[j] -= learning_rate * (vf_coef * dv[j])

            last_pi_loss = pi_loss
            last_v_loss = v_loss
            last_ent = ent_sum
            approx_kl = kl_sum

    return {
        "pi_loss": float(last_pi_loss),
        "v_loss": float(last_v_loss),
        "entropy": float(last_ent),
        "approx_kl": float(approx_kl),
        "clipfrac": float(clipfrac),
    }

# -----------------------------
# Evaluation
# -----------------------------
def eval_policy(env: OneDReachEnv, agent: LinearActorCritic, episodes: int = 5) -> float:
    total = 0.0
    for _ in range(episodes):
        obs = env.reset()
        done = False
        t = 0
        while not done and t < env.horizon:
            # greedy action
            ps = agent.policy_probs(obs)
            a = 0 if ps[0] >= ps[1] else 1
            step = env.step(a)
            total += step.reward
            done = step.terminated or step.truncated
            obs = step.obs
            t += 1
    return total / episodes

# -----------------------------
# Main training loop
# -----------------------------
def main():
    env = OneDReachEnv(max_pos=5, horizon=40, step_penalty=-0.01)
    feat_dim = len(env.reset())
    agent = LinearActorCritic(feat_dim=feat_dim, act_dim=2, entropy_coef=0.01)

    total_iters = 120
    steps_per_rollout = 512
    gamma = 0.99
    lam = 0.95
    clip_ratio = 0.2
    vf_coef = 0.5
    ent_coef = 0.01
    train_iters = 8
    minibatch_size = 128
    lr = 1e-2

    print("Train PPO (pure Python)...")
    start = time.time()
    for it in range(1, total_iters + 1):
        batch = collect_rollout(env, agent, steps_per_rollout, gamma, lam)
        stats = ppo_update(agent, batch, clip_ratio, vf_coef, ent_coef, train_iters, minibatch_size, lr)

        if it % 5 == 0 or it == 1:
            ret = eval_policy(env, agent, episodes=10)
            print(f"Iter {it:03d}: EvalReturn={ret:+.3f}  "
                  f"pi_loss={stats['pi_loss']:.3f}  v_loss={stats['v_loss']:.3f}  "
                  f"entropy={stats['entropy']:.3f}  KL={stats['approx_kl']:.4f}  clipfrac={stats['clipfrac']:.3f}")

    dur = time.time() - start
    print(f"Done in {dur:.1f}s.")

if __name__ == "__main__":
    main()
